# l1-primitive-barrier
GPU programming models typically support a hiearchcal execution abstraction which reflects the hiearchical nature of the hardware. where individual threads reside eat the base level. These threads are organized into subgroups, which are further organized into workgroups. 

Threads within the same subgroup typically execute on the same SIMD execution group (which AMD calls wavefronts and NVIDIA calls warps) [1].
Because of this assumption, threads within a subgroup can use fast private memory regions for inter-thread communication.
Subgroups are a software abstraction, meaning this direct mapping to SIMD hardware may not always exist. Because of this, threads within a workgroup could potentially span multiple SEGs (or not fully saturate a single one), necessitating the existence of a subgroup barrier primitive in order to synchronize execution and memory gurantees.

Subgroups are further organized within workgroups, and can employ local memory regions for communication between threads within the same workgroup.









# References
[1] Sorensen, “Inter-Workgroup Barrier Synchronisation on Graphics Processing Units.”