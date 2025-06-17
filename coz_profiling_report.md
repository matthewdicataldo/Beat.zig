# ZigPulse COZ Profiling Report

Generated: Mon Jun 16 16:37:19 PDT 2025
Build mode: ReleaseSafe

## Key Findings

### Progress Points
- **zigpulse_task_completed**: Measures overall throughput
- **zigpulse_task_execution**: Measures task latency
- **zigpulse_task_stolen**: Measures work-stealing activity
- **zigpulse_worker_idle**: Measures load balancing

## Analysis Instructions

1. Upload zigpulse_profile.coz to https://plasma-umass.org/coz/
2. Look for lines with positive virtual speedup
3. Focus optimization efforts on code with highest impact

## Common Bottlenecks

- **Task Submission**: If submit() shows high impact
- **Work Stealing**: If steal operations show high impact
- **Memory Management**: If allocation/free shows high impact
- **Queue Contention**: If mutex operations show high impact

## Next Steps

1. Identify hotspots from COZ visualization
2. Implement targeted optimizations
3. Re-run profiling to verify improvements
