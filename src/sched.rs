//! Cooperative goroutine scheduler and channels.
//!
//! fusevm's dispatch loop runs one VM to completion; this module layers a
//! green-thread scheduler on top so a frontend (e.g. Go) can express
//! goroutines and channels. Each goroutine is its own [`VM`] instance sharing
//! the program `Chunk` and the frontend's thread-local heap; the scheduler
//! owns the channel table and a run queue.
//!
//! The mechanism reuses the same "op raises a request, halts, driver reads it
//! after `run()`" pattern as `Op::AwkSignal` — see [`crate::op::Op::Go`] and the
//! channel ops. A goroutine/channel op stashes a [`SchedReq`] in the VM and
//! halts; the scheduler reads it via [`VM::take_sched`], services it (delivering
//! any result by pushing onto the target VM's stack), and resumes that VM or
//! switches to another. Because the op has already advanced the VM's `ip`, a
//! resumed VM continues *past* the op — the scheduler never rewinds or
//! re-executes, so no result is lost.
//!
//! Channels are host-opaque `Value::Int(id)` handles into the scheduler's
//! channel table. Send/receive follow CSP semantics: a buffered channel holds up
//! to `cap` values; an unbuffered channel hands a value directly from a blocked
//! sender to a blocked receiver. When every goroutine is blocked, the scheduler
//! reports a deadlock (Go's "all goroutines are asleep").
//!
//! This is single-threaded and cooperative: goroutines yield only at channel
//! operations (and completion). That is a faithful model for channel-driven Go
//! programs — the common shape — without the data races an OS-thread model would
//! introduce over the frontend's thread-local heap.

use crate::{Frame, VMResult, Value, VM};
use std::collections::{HashSet, VecDeque};

/// A scheduling request a goroutine/channel op raises in the VM (`self.sched`),
/// read by the [`Scheduler`] after `run()` returns.
#[derive(Debug, Clone)]
pub enum SchedReq {
    /// `go f(args)` — spawn a goroutine running sub `name_idx` with `args`.
    Go { name_idx: u16, args: Vec<Value> },
    /// `make(chan T, cap)` — allocate a channel; its id is pushed to the caller.
    Make { cap: usize },
    /// `ch <- val` — send `val` on channel `ch` (may block).
    Send { ch: i64, val: Value },
    /// `<-ch` — receive from channel `ch`, pushing the value (may block).
    Recv { ch: i64 },
    /// `v, ok := <-ch` — receive from channel `ch`, pushing `[value, ok]` with
    /// `ok` on top (`1` = a real value was received, `0` = the channel is closed
    /// and drained, so `value` is the zero value). May block; a receive that
    /// blocks is delivered the same pair when it is woken.
    ///
    /// The distinction [`SchedReq::Recv`] cannot express: a plain receive yields
    /// the element-type zero for both "a sender sent 0" and "closed and
    /// drained", which makes `for v := range ch` unimplementable on it.
    RecvOk { ch: i64 },
    /// `close(ch)` — close channel `ch`.
    Close { ch: i64 },
    /// A `select` over channel operations: pick a ready case (else the default,
    /// else block until one is ready). Pushes `[recv_value, case_index]`.
    Select {
        cases: Vec<SelectCase>,
        has_default: bool,
    },
}

/// One communication clause of a `select`.
#[derive(Debug, Clone)]
pub struct SelectCase {
    /// True for a receive (`<-ch`), false for a send (`ch <- val`).
    pub recv: bool,
    /// The channel id.
    pub ch: i64,
    /// The value to send (unused for a receive).
    pub val: Value,
}

/// Why the scheduler stopped without the main goroutine finishing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedError {
    /// Every live goroutine is blocked on a channel — no progress is possible.
    Deadlock,
    /// A goroutine faulted (a channel misuse, or a VM runtime error).
    Panic(String),
}

impl std::fmt::Display for SchedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedError::Deadlock => {
                write!(f, "fatal error: all goroutines are asleep - deadlock!")
            }
            SchedError::Panic(m) => write!(f, "{m}"),
        }
    }
}

/// A channel: a bounded value buffer plus queues of goroutines blocked sending
/// (with their pending value) or receiving.
struct Channel {
    cap: usize,
    buf: VecDeque<Value>,
    closed: bool,
    send_q: VecDeque<(usize, Value)>,
    /// Goroutines blocked receiving, each with the flag "wants the `ok` result
    /// too" (a [`SchedReq::RecvOk`] receive). A plain [`SchedReq::Recv`] parks
    /// with `false` and is delivered the bare value, exactly as before.
    recv_q: VecDeque<(usize, bool)>,
}

impl Channel {
    fn new(cap: usize) -> Self {
        Channel {
            cap,
            buf: VecDeque::new(),
            closed: false,
            send_q: VecDeque::new(),
            recv_q: VecDeque::new(),
        }
    }
}

/// The cooperative goroutine scheduler. Construct with [`Scheduler::new`], then
/// [`Scheduler::run`].
pub struct Scheduler<F: FnMut() -> VM> {
    /// Fresh, fully-configured VM factory (same chunk + builtins + hooks) used to
    /// materialize each spawned goroutine. The scheduler positions it at the
    /// goroutine's entry.
    make_vm: F,
    vms: Vec<VM>,
    ready: VecDeque<usize>,
    blocked: HashSet<usize>,
    done: HashSet<usize>,
    chans: Vec<Channel>,
    /// Goroutines blocked in a `select`, with their cases; re-checked whenever a
    /// channel changes state (see [`Scheduler::recheck_selects`]).
    select_waiters: Vec<(usize, Vec<SelectCase>)>,
    /// The zero value a receive yields on a closed, empty channel. Frontends set
    /// this to the element type's zero (`Value::Int(0)` by default).
    recv_zero: Value,
}

impl<F: FnMut() -> VM> Scheduler<F> {
    /// Create a scheduler whose goroutines are built by `make_vm`.
    pub fn new(make_vm: F) -> Self {
        Scheduler {
            make_vm,
            vms: Vec::new(),
            ready: VecDeque::new(),
            blocked: HashSet::new(),
            done: HashSet::new(),
            chans: Vec::new(),
            select_waiters: Vec::new(),
            recv_zero: Value::Int(0),
        }
    }

    /// Set the value a receive on a closed, empty channel yields (default `0`).
    pub fn with_recv_zero(mut self, zero: Value) -> Self {
        self.recv_zero = zero;
        self
    }

    /// Drive `main_vm` (goroutine 0) and every goroutine it spawns to completion.
    /// Returns when the main goroutine finishes (Go semantics: the program exits
    /// when `main` returns) or on deadlock / panic.
    pub fn run(mut self, main_vm: VM) -> Result<(), SchedError> {
        self.vms.push(main_vm);
        self.ready.push_back(0);
        self.drive()
    }

    /// Like [`Scheduler::run`], but afterward returns the value of `global` from
    /// the main goroutine (`None` if the program has no such global). Lets a
    /// frontend read `main`'s result out of a scheduled run.
    pub fn run_capturing(mut self, main_vm: VM, global: &str) -> Result<Option<Value>, SchedError> {
        self.vms.push(main_vm);
        self.ready.push_back(0);
        self.drive()?;
        let main = &self.vms[0];
        let val = main
            .chunk
            .names
            .iter()
            .position(|n| n == global)
            .and_then(|i| main.globals.get(i).cloned());
        Ok(val)
    }

    /// The scheduling loop over the ready queue.
    fn drive(&mut self) -> Result<(), SchedError> {
        while let Some(gid) = self.ready.pop_front() {
            self.vms[gid].clear_halt();
            // `VM::run` pops the top of the value stack on exit to return a
            // result. When a goroutine/channel op halts the VM mid-expression,
            // that popped value is live data the op left below its operands —
            // capture it and restore it before servicing the park.
            let popped = match self.vms[gid].run() {
                VMResult::Error(e) => return Err(SchedError::Panic(e)),
                VMResult::Ok(v) => Some(v),
                VMResult::Halted => None,
            };
            match self.vms[gid].take_sched() {
                Some(req) => {
                    if let Some(v) = popped {
                        self.vms[gid].stack.push(v);
                    }
                    self.service(gid, req)?;
                }
                None => {
                    // The goroutine finished (or halted for a frontend reason);
                    // `popped` is its final result, which the caller ignores.
                    self.done.insert(gid);
                    if gid == 0 {
                        // main returned — the program exits.
                        return Ok(());
                    }
                }
            }

            if self.ready.is_empty() && !self.blocked.is_empty() {
                return Err(SchedError::Deadlock);
            }
        }
        Ok(())
    }

    /// Service one scheduling request from goroutine `gid`.
    fn service(&mut self, gid: usize, req: SchedReq) -> Result<(), SchedError> {
        match req {
            SchedReq::Make { cap } => {
                let id = self.chans.len();
                self.chans.push(Channel::new(cap));
                self.vms[gid].stack.push(Value::Int(id as i64));
                self.ready.push_back(gid);
            }
            SchedReq::Go { name_idx, args } => {
                let ngid = self.vms.len();
                let mut vm = (self.make_vm)();
                self.position_goroutine(&mut vm, name_idx, &args)?;
                self.vms.push(vm);
                self.ready.push_back(ngid);
                // The spawner continues immediately after `go`.
                self.ready.push_back(gid);
            }
            SchedReq::Send { ch, val } => {
                if self.try_send(ch, &val)? {
                    self.ready.push_back(gid);
                    self.recheck_selects()?;
                } else {
                    self.chan_mut(ch)?.send_q.push_back((gid, val));
                    self.blocked.insert(gid);
                }
            }
            SchedReq::Recv { ch } => {
                if let Some((v, _ok)) = self.try_recv(ch)? {
                    self.vms[gid].stack.push(v);
                    self.ready.push_back(gid);
                    self.recheck_selects()?;
                } else {
                    self.chan_mut(ch)?.recv_q.push_back((gid, false));
                    self.blocked.insert(gid);
                }
            }
            SchedReq::RecvOk { ch } => {
                if let Some((v, ok)) = self.try_recv(ch)? {
                    self.vms[gid].stack.push(v);
                    self.vms[gid].stack.push(Value::Int(ok as i64));
                    self.ready.push_back(gid);
                    self.recheck_selects()?;
                } else {
                    self.chan_mut(ch)?.recv_q.push_back((gid, true));
                    self.blocked.insert(gid);
                }
            }
            SchedReq::Select { cases, has_default } => {
                if let Some((idx, rv)) = self.try_select(&cases)? {
                    self.deliver_select(gid, idx, rv);
                    self.recheck_selects()?;
                } else if has_default {
                    // The `default` case: sentinel index = number of real cases.
                    self.deliver_select(gid, cases.len(), Value::Undef);
                } else {
                    self.select_waiters.push((gid, cases));
                    self.blocked.insert(gid);
                }
            }
            SchedReq::Close { ch } => {
                let c = self.chan_mut(ch)?;
                c.closed = true;
                // Wake every blocked receiver with the zero value — and, for a
                // `RecvOk` receiver, `ok = 0`: it blocked on an empty channel,
                // so the close is what freed it and there is no value.
                let woken: Vec<(usize, bool)> = c.recv_q.drain(..).collect();
                for (r, wants_ok) in woken {
                    self.vms[r].stack.push(self.recv_zero.clone());
                    if wants_ok {
                        self.vms[r].stack.push(Value::Int(0));
                    }
                    self.wake(r);
                }
                self.ready.push_back(gid);
                self.recheck_selects()?;
            }
        }
        Ok(())
    }

    /// Push a select outcome (`[recv_value, case_index]`) onto goroutine `gid`
    /// and make it runnable. The compiled `select` reads the index to jump to
    /// the winning case and the value to bind a `case v := <-ch`.
    fn deliver_select(&mut self, gid: usize, idx: usize, rv: Value) {
        self.vms[gid].stack.push(rv);
        self.vms[gid].stack.push(Value::Int(idx as i64));
        self.wake(gid);
    }

    /// Attempt a receive on `ch` without blocking: `Some((value, ok))` if it
    /// completed (performing the receive and waking any blocked sender), `None`
    /// if it would block.
    ///
    /// `ok` is `false` only for the closed-and-drained case, where `value` is the
    /// zero value rather than anything a sender produced. Callers serving a plain
    /// [`SchedReq::Recv`] discard it and behave exactly as before.
    fn try_recv(&mut self, ch: i64) -> Result<Option<(Value, bool)>, SchedError> {
        enum O {
            V(Value),
            Wake(usize, Value),
            Zero,
            No,
        }
        let o = {
            let c = self.chan_mut(ch)?;
            if let Some(v) = c.buf.pop_front() {
                if let Some((s, sv)) = c.send_q.pop_front() {
                    c.buf.push_back(sv);
                    O::Wake(s, v)
                } else {
                    O::V(v)
                }
            } else if let Some((s, sv)) = c.send_q.pop_front() {
                O::Wake(s, sv)
            } else if c.closed {
                O::Zero
            } else {
                O::No
            }
        };
        Ok(match o {
            O::V(v) => Some((v, true)),
            O::Wake(s, v) => {
                self.wake(s);
                Some((v, true))
            }
            O::Zero => Some((self.recv_zero.clone(), false)),
            O::No => None,
        })
    }

    /// Attempt a send on `ch` without blocking: `true` if it completed (delivered
    /// to a blocked receiver or buffered), `false` if it would block. Panics on a
    /// closed channel.
    fn try_send(&mut self, ch: i64, val: &Value) -> Result<bool, SchedError> {
        enum O {
            Deliver(usize, bool),
            Buffered,
            No,
        }
        let o = {
            let c = self.chan_mut(ch)?;
            if c.closed {
                return Err(SchedError::Panic(
                    "panic: send on closed channel".to_string(),
                ));
            }
            if let Some((r, wants_ok)) = c.recv_q.pop_front() {
                O::Deliver(r, wants_ok)
            } else if c.buf.len() < c.cap {
                c.buf.push_back(val.clone());
                O::Buffered
            } else {
                O::No
            }
        };
        Ok(match o {
            O::Deliver(r, wants_ok) => {
                self.vms[r].stack.push(val.clone());
                if wants_ok {
                    // A real value crossed the channel, so `ok` is true.
                    self.vms[r].stack.push(Value::Int(1));
                }
                self.wake(r);
                true
            }
            O::Buffered => true,
            O::No => false,
        })
    }

    /// Try each `select` case in order; perform the first ready one and return
    /// `(case index, received value)` (the value is nil for a send/default).
    fn try_select(&mut self, cases: &[SelectCase]) -> Result<Option<(usize, Value)>, SchedError> {
        for (i, c) in cases.iter().enumerate() {
            if c.recv {
                // `select` delivers `[value, case_index]`; the closed flag has no
                // slot in that shape, so it is discarded here (unchanged
                // behaviour — a `case v, ok := <-ch` would need its own op).
                if let Some((v, _ok)) = self.try_recv(c.ch)? {
                    return Ok(Some((i, v)));
                }
            } else if self.try_send(c.ch, &c.val)? {
                return Ok(Some((i, Value::Undef)));
            }
        }
        Ok(None)
    }

    /// Re-attempt every blocked `select` after a channel-state change; any whose
    /// case is now ready proceeds. Loops until no waiting select can progress (a
    /// wakeup may enable another).
    fn recheck_selects(&mut self) -> Result<(), SchedError> {
        loop {
            let mut progressed = false;
            let waiters = std::mem::take(&mut self.select_waiters);
            let mut still = Vec::new();
            for (gid, cases) in waiters {
                match self.try_select(&cases)? {
                    Some((idx, rv)) => {
                        self.deliver_select(gid, idx, rv);
                        progressed = true;
                    }
                    None => still.push((gid, cases)),
                }
            }
            self.select_waiters = still;
            if !progressed {
                break;
            }
        }
        Ok(())
    }

    /// Move a blocked goroutine back to the ready queue.
    fn wake(&mut self, gid: usize) {
        self.blocked.remove(&gid);
        self.ready.push_back(gid);
    }

    fn chan_mut(&mut self, ch: i64) -> Result<&mut Channel, SchedError> {
        self.chans
            .get_mut(ch as usize)
            .ok_or_else(|| SchedError::Panic(format!("panic: invalid channel {ch}")))
    }

    /// Position a fresh goroutine VM to start executing sub `name_idx` with
    /// `args`, returning to a terminal ip when the sub returns (so `run()` ends).
    fn position_goroutine(
        &self,
        vm: &mut VM,
        name_idx: u16,
        args: &[Value],
    ) -> Result<(), SchedError> {
        let entry = vm.chunk.find_sub(name_idx).ok_or_else(|| {
            SchedError::Panic(format!("panic: goroutine target sub {name_idx} not found"))
        })?;
        let base = vm.stack.len();
        for a in args {
            vm.stack.push(a.clone());
        }
        vm.frames.push(Frame {
            return_ip: vm.chunk.ops.len(), // returning ends the goroutine's run()
            stack_base: base,
            slots: Vec::new(),
            entry_ip: Some(entry),
        });
        vm.ip = entry;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Chunk, ChunkBuilder, Op};

    /// Run `chunk` under a fresh scheduler and read back `global` from main.
    fn run(chunk: Chunk, global: &str) -> Result<Option<Value>, SchedError> {
        let fc = chunk.clone();
        Scheduler::new(move || VM::new(fc.clone())).run_capturing(VM::new(chunk), global)
    }

    #[test]
    fn producer_consumer_over_unbuffered_channel() {
        // main:
        //   ch := make(chan int)          // cap 0 (unbuffered)
        //   go producer(ch)
        //   sum := <-ch + <-ch + <-ch     // synchronizes with the sender
        // producer(ch): send 1, 2, 3
        let mut b = ChunkBuilder::new();
        let prod = b.add_name("producer");
        let ch = b.add_name("ch");
        let sum = b.add_name("sum");

        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::ChanMake, 1);
        b.emit(Op::SetVar(ch), 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::Go(prod, 1), 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanRecv, 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanRecv, 1);
        b.emit(Op::Add, 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanRecv, 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetVar(sum), 1);
        let skip = b.emit(Op::Jump(0), 1);

        let entry = b.current_pos();
        b.add_sub_entry(prod, entry);
        b.emit(Op::SetSlot(0), 2); // bind the channel arg
        for v in 1..=3 {
            b.emit(Op::GetSlot(0), 2);
            b.emit(Op::LoadInt(v), 2);
            b.emit(Op::ChanSend, 2);
        }
        b.emit(Op::LoadUndef, 2);
        b.emit(Op::ReturnValue, 2);
        b.patch_jump(skip, b.current_pos());

        assert_eq!(run(b.build(), "sum"), Ok(Some(Value::Int(6))));
    }

    #[test]
    fn buffered_channel_holds_values_without_a_receiver() {
        // ch := make(chan int, 2); ch <- 10; ch <- 20; sum := <-ch + <-ch
        // The two sends do not block (buffer capacity 2), all in one goroutine.
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        let sum = b.add_name("sum");
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::ChanMake, 1);
        b.emit(Op::SetVar(ch), 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::LoadInt(10), 1);
        b.emit(Op::ChanSend, 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::LoadInt(20), 1);
        b.emit(Op::ChanSend, 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanRecv, 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanRecv, 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetVar(sum), 1);
        assert_eq!(run(b.build(), "sum"), Ok(Some(Value::Int(30))));
    }

    #[test]
    fn select_picks_the_ready_case() {
        // ch1 (unbuffered, no sender) and ch2 (buffered, holds 7).
        //   select { case <-ch1: ; case v := <-ch2: sum = v }
        // must pick case 1 (ch2 ready) and receive 7.
        let mut b = ChunkBuilder::new();
        let ch1 = b.add_name("ch1");
        let ch2 = b.add_name("ch2");
        let sum = b.add_name("sum");
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::ChanMake, 1);
        b.emit(Op::SetVar(ch1), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::ChanMake, 1);
        b.emit(Op::SetVar(ch2), 1);
        b.emit(Op::GetVar(ch2), 1);
        b.emit(Op::LoadInt(7), 1);
        b.emit(Op::ChanSend, 1);
        // case 0: recv ch1 — [ch1, is_recv=1, val=0]
        b.emit(Op::GetVar(ch1), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(0), 1);
        // case 1: recv ch2 — [ch2, is_recv=1, val=0]
        b.emit(Op::GetVar(ch2), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::Select(2, 0), 1);
        // stack: [recv_value, case_index]; both cases here store the value, so
        // drop the index and keep the value.
        b.emit(Op::Pop, 1);
        b.emit(Op::SetVar(sum), 1);
        assert_eq!(run(b.build(), "sum"), Ok(Some(Value::Int(7))));
    }

    #[test]
    fn select_default_when_no_case_ready() {
        // An empty channel with a `default` case takes the default (index = 1).
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        let idx = b.add_name("idx");
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::ChanMake, 1);
        b.emit(Op::SetVar(ch), 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::Select(1, 1), 1); // 1 case, has_default
        b.emit(Op::SetVar(idx), 1); // index (default sentinel = 1)
        b.emit(Op::Pop, 1); // drop the recv value
        assert_eq!(run(b.build(), "idx"), Ok(Some(Value::Int(1))));
    }

    #[test]
    fn repro_closed_drained_recv_is_indistinguishable_from_a_received_zero() {
        // ch := make(chan int, 2); ch <- 10; ch <- 20; close(ch)
        // <-ch, <-ch drain 10 and 20; the third receive yields the zero value.
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        let out = b.add_name("out");
        b.emit(Op::LoadInt(2), 1);
        b.emit(Op::ChanMake, 1);
        b.emit(Op::SetVar(ch), 1);
        for v in [10, 20] {
            b.emit(Op::GetVar(ch), 1);
            b.emit(Op::LoadInt(v), 1);
            b.emit(Op::ChanSend, 1);
        }
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanClose, 1);
        for _ in 0..2 {
            b.emit(Op::GetVar(ch), 1);
            b.emit(Op::ChanRecv, 1);
            b.emit(Op::Pop, 1);
        }
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanRecv, 1);
        b.emit(Op::SetVar(out), 1);
        let drained = run(b.build(), "out");

        // An OPEN channel carrying a legitimate 0 yields the same thing.
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        let out = b.add_name("out");
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::ChanMake, 1);
        b.emit(Op::SetVar(ch), 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::ChanSend, 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanRecv, 1);
        b.emit(Op::SetVar(out), 1);
        let real_zero = run(b.build(), "out");

        assert_eq!(drained, Ok(Some(Value::Int(0))));
        assert_eq!(real_zero, Ok(Some(Value::Int(0))));
        assert_eq!(
            drained, real_zero,
            "Op::ChanRecv cannot report closed-and-drained; use Op::ChanRecvOk"
        );
    }

    /// Emit `ch := make(chan int, cap)` into `b`, binding it to name `ch`.
    fn make_chan(b: &mut ChunkBuilder, ch: u16, cap: i64) {
        b.emit(Op::LoadInt(cap), 1);
        b.emit(Op::ChanMake, 1);
        b.emit(Op::SetVar(ch), 1);
    }

    /// Emit `v, ok := <-ch`, binding the pair to names `v` and `ok`. Pinning the
    /// pop order here is the point: `Op::ChanRecvOk` pushes the value first and
    /// `ok` on top, so `ok` is stored first.
    fn recv_ok(b: &mut ChunkBuilder, ch: u16, v: u16, ok: u16) {
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanRecvOk, 1);
        b.emit(Op::SetVar(ok), 1);
        b.emit(Op::SetVar(v), 1);
    }

    #[test]
    fn recv_ok_reports_true_for_a_buffered_value_and_false_once_drained() {
        // ch := make(chan int, 2); ch <- 10; ch <- 20; close(ch)
        // Three two-value receives: (10,1), (20,1), (0,0).
        // Encoded as one global per field so the drain ORDER is pinned, not just
        // the final state.
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        let (v0, k0) = (b.add_name("v0"), b.add_name("k0"));
        let (v1, k1) = (b.add_name("v1"), b.add_name("k1"));
        let (v2, k2) = (b.add_name("v2"), b.add_name("k2"));
        make_chan(&mut b, ch, 2);
        for v in [10, 20] {
            b.emit(Op::GetVar(ch), 1);
            b.emit(Op::LoadInt(v), 1);
            b.emit(Op::ChanSend, 1);
        }
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanClose, 1);
        recv_ok(&mut b, ch, v0, k0);
        recv_ok(&mut b, ch, v1, k1);
        recv_ok(&mut b, ch, v2, k2);
        let chunk = b.build();

        for (name, want) in [
            ("v0", 10),
            ("k0", 1),
            ("v1", 20),
            ("k1", 1),
            ("v2", 0),
            ("k2", 0),
        ] {
            assert_eq!(
                run(chunk.clone(), name),
                Ok(Some(Value::Int(want))),
                "{name}"
            );
        }
    }

    #[test]
    fn recv_ok_on_a_channel_closed_with_no_values_is_false() {
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        let (v, k) = (b.add_name("v"), b.add_name("k"));
        make_chan(&mut b, ch, 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanClose, 1);
        recv_ok(&mut b, ch, v, k);
        let chunk = b.build();
        assert_eq!(run(chunk.clone(), "v"), Ok(Some(Value::Int(0))));
        assert_eq!(run(chunk, "k"), Ok(Some(Value::Int(0))));
    }

    #[test]
    fn recv_ok_of_a_real_zero_from_an_open_channel_is_true() {
        // The case that rules out making `recv_zero` a sentinel: 0 is a
        // legitimate value on a `chan int`, and must read as received.
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        let (v, k) = (b.add_name("v"), b.add_name("k"));
        make_chan(&mut b, ch, 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::ChanSend, 1);
        recv_ok(&mut b, ch, v, k);
        let chunk = b.build();
        assert_eq!(run(chunk.clone(), "v"), Ok(Some(Value::Int(0))));
        assert_eq!(run(chunk, "k"), Ok(Some(Value::Int(1))));
    }

    #[test]
    fn recv_ok_on_an_open_empty_channel_blocks() {
        // Nobody ever sends or closes: the two-value receive must park like a
        // plain one, not fall through with `ok = 0`.
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        let (v, k) = (b.add_name("v"), b.add_name("k"));
        make_chan(&mut b, ch, 1);
        recv_ok(&mut b, ch, v, k);
        assert_eq!(run(b.build(), "v"), Err(SchedError::Deadlock));
    }

    #[test]
    fn a_blocked_recv_ok_is_woken_true_by_a_sender_and_false_by_a_close() {
        // main blocks on an unbuffered two-value receive; the goroutine sends 5
        // (→ ok = 1), then main blocks again and the goroutine closes (→ ok = 0).
        let mut b = ChunkBuilder::new();
        let sender = b.add_name("sender");
        let ch = b.add_name("ch");
        let (v0, k0) = (b.add_name("v0"), b.add_name("k0"));
        let (v1, k1) = (b.add_name("v1"), b.add_name("k1"));
        make_chan(&mut b, ch, 0);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::Go(sender, 1), 1);
        recv_ok(&mut b, ch, v0, k0);
        recv_ok(&mut b, ch, v1, k1);
        let skip = b.emit(Op::Jump(0), 1);

        let entry = b.current_pos();
        b.add_sub_entry(sender, entry);
        b.emit(Op::SetSlot(0), 2);
        b.emit(Op::GetSlot(0), 2);
        b.emit(Op::LoadInt(5), 2);
        b.emit(Op::ChanSend, 2);
        b.emit(Op::GetSlot(0), 2);
        b.emit(Op::ChanClose, 2);
        b.emit(Op::LoadUndef, 2);
        b.emit(Op::ReturnValue, 2);
        b.patch_jump(skip, b.current_pos());
        let chunk = b.build();

        for (name, want) in [("v0", 5), ("k0", 1), ("v1", 0), ("k1", 0)] {
            assert_eq!(
                run(chunk.clone(), name),
                Ok(Some(Value::Int(want))),
                "{name}"
            );
        }
    }

    #[test]
    fn range_over_a_closed_channel_yields_every_value_then_stops() {
        // The defect this op exists for:
        //   sum, n := 0, 0
        //   for v := range ch { sum += v; n++ }
        // over a buffered channel holding 1, 2, 3 that is then closed. `range`
        // lowers to `for { v, ok := <-ch; if !ok { break }; ... }`.
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        let (v, k) = (b.add_name("v"), b.add_name("k"));
        let sum = b.add_name("sum");
        let n = b.add_name("n");
        make_chan(&mut b, ch, 3);
        for x in 1..=3 {
            b.emit(Op::GetVar(ch), 1);
            b.emit(Op::LoadInt(x), 1);
            b.emit(Op::ChanSend, 1);
        }
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanClose, 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetVar(sum), 1);
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::SetVar(n), 1);

        let top = b.current_pos();
        recv_ok(&mut b, ch, v, k);
        b.emit(Op::GetVar(k), 1);
        let exit = b.emit(Op::JumpIfFalse(0), 1);
        b.emit(Op::GetVar(sum), 1);
        b.emit(Op::GetVar(v), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetVar(sum), 1);
        b.emit(Op::GetVar(n), 1);
        b.emit(Op::LoadInt(1), 1);
        b.emit(Op::Add, 1);
        b.emit(Op::SetVar(n), 1);
        let back = b.emit(Op::Jump(0), 1);
        b.patch_jump(back, top);
        b.patch_jump(exit, b.current_pos());
        let chunk = b.build();

        assert_eq!(run(chunk.clone(), "n"), Ok(Some(Value::Int(3))));
        assert_eq!(run(chunk, "sum"), Ok(Some(Value::Int(6))));
    }

    #[test]
    fn recv_ok_leaves_plain_recv_alone_on_the_same_channel() {
        // Mixing the two receives on one channel: the plain form must still push
        // exactly one value (no stray `ok` on the stack desyncing the caller).
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        let (v, k) = (b.add_name("v"), b.add_name("k"));
        let plain = b.add_name("plain");
        make_chan(&mut b, ch, 2);
        for x in [7, 9] {
            b.emit(Op::GetVar(ch), 1);
            b.emit(Op::LoadInt(x), 1);
            b.emit(Op::ChanSend, 1);
        }
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanRecv, 1);
        b.emit(Op::SetVar(plain), 1);
        recv_ok(&mut b, ch, v, k);
        let chunk = b.build();
        assert_eq!(run(chunk.clone(), "plain"), Ok(Some(Value::Int(7))));
        assert_eq!(run(chunk.clone(), "v"), Ok(Some(Value::Int(9))));
        assert_eq!(run(chunk, "k"), Ok(Some(Value::Int(1))));
    }

    #[test]
    fn recv_ok_uses_the_frontends_recv_zero_when_drained() {
        // A non-integer element type: the drained value is the configured zero,
        // and `ok` is what tells the frontend it is not real data.
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        let (v, k) = (b.add_name("v"), b.add_name("k"));
        make_chan(&mut b, ch, 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanClose, 1);
        recv_ok(&mut b, ch, v, k);
        let chunk = b.build();
        let fc = chunk.clone();
        let got = Scheduler::new(move || VM::new(fc.clone()))
            .with_recv_zero(Value::str(""))
            .run_capturing(VM::new(chunk.clone()), "v");
        assert_eq!(got, Ok(Some(Value::str(""))));

        let fc = chunk.clone();
        let got = Scheduler::new(move || VM::new(fc.clone()))
            .with_recv_zero(Value::str(""))
            .run_capturing(VM::new(chunk), "k");
        assert_eq!(got, Ok(Some(Value::Int(0))));
    }

    #[test]
    fn deadlock_is_detected() {
        // main receives on a channel nobody ever sends to.
        let mut b = ChunkBuilder::new();
        let ch = b.add_name("ch");
        b.emit(Op::LoadInt(0), 1);
        b.emit(Op::ChanMake, 1);
        b.emit(Op::SetVar(ch), 1);
        b.emit(Op::GetVar(ch), 1);
        b.emit(Op::ChanRecv, 1);
        b.emit(Op::Pop, 1);
        assert_eq!(run(b.build(), "x"), Err(SchedError::Deadlock));
    }
}
