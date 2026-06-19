# rocm.md

**⚠️ CUDA/NVIDIA Path Status: DEPRECATED**
The CUDA (NVIDIA) code path is currently **untested and may not compile**. Due to lack of access to NVIDIA hardware for testing, the CUDA backend cannot be verified. The CUDA code remains in the codebase but is not actively maintained. Use the HIP (AMD) or CPU backend instead.

## Goal

Write correct, maintainable HIP/ROCm code.

Priority:

```text
Correctness
> Memory locality
> Throughput
> Occupancy
> Micro-optimization
```

---

## Porting Rules

* Preserve semantics.
* Preserve indexing.
* Preserve synchronization.
* Preserve memory layout until profiling justifies change.
* Port first. Optimize later.
* One change class at time.

Forbidden:

* Port + refactor.
* Port + optimization.
* Port + algorithm rewrite.

---

## Execution Model

Think wavefronts, not threads.

Questions:

* Coalesced access?
* Divergence?
* Register pressure?
* LDS reuse?
* Occupancy sufficient to hide latency?

---

## Memory Rules

Assume memory-bound until profiling proves otherwise.

Priority:

```text
Registers
> LDS
> Cache
> Global memory
```

Rules:

* Load once.
* Reuse many times.
* Write once.
* Keep neighboring lanes on neighboring addresses.
* Prefer contiguous reads/writes.
* Minimize atomics.
* Minimize memory traffic before optimizing math.

---

## Divergence Rules

Avoid divergent hot paths.

Bad:

```cpp
if(condition_per_lane)
```

Prefer:

* uniform branches
* predication
* work partitioning
* separate kernels

---

## Occupancy Rules

Occupancy ≠ performance.

Watch:

* VGPR usage
* LDS usage
* spills
* wave residency

Higher occupancy only matters if latency hidden better.

---

## Synchronization Rules

Use smallest correct scope.

* No barrier without reason.
* No atomics without reason.
* No host sync without dependency.

---

## Agent Rules

Do not:

* invent optimizations
* change numerical behavior
* remove barriers blindly
* change launch geometry without reason
* replace clear code with clever code

Do:

* explain non-trivial changes
* keep generated code compilable
* preserve behavior
* profile-driven optimize

---

## Profiling Rules

Never claim performance improvement without measurement.

Check:

* kernel time
* memory throughput
* occupancy
* VGPR pressure
* LDS pressure
* divergence
* atomic contention

---

## Cardinal Sins

* Random memory access
* Divergent hot paths
* Excessive atomics
* Register spills
* Tiny kernels
* Synchronization abuse
* Premature optimization
* CPU-style thinking

---

## Default Assumption

Problem memory-bound until profiler says otherwise.
