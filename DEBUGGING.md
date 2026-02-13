# Size 40 FFT Bug Investigation

## Summary
- Only size 40 fails among all tested highly-composite sizes
- DC component shows only first 5 elements are summed (0+1+2+3+4=10 instead of 780)
- Delta function test shows output at indices 0,8,16,24,32 (5 values spaced by 8)
- Size 40 = 5×8, and 8 is the span for radix-5

## Hypothesis
The radix-5 butterfly is being applied, but the recursive calls for the 4×2 decomposition are either:
1. Not being made
2. Processing wrong data
3. Writing results to wrong locations

## Next Step  
Add temporary debug logging to mixedRadixRecursivePingPongComplex128 to trace:
- Each recursive call depth
- Values of n, stride, step, radices at each level
- Which code path (base case vs recursion) is taken
