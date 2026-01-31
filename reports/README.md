"Cost calibration" files are of the form "cc-<ll>-by-<mod>.txt", where<ll> is the length limit for punchlines, and <mod> is the name of the model.

```
1218 47 ==> (  1000/12.6%): [+] 589 (0.6) (94.404% 0.098% 5.498%) 23s 
```

...means that strip 1218 (https://www.qwantz.com/index.php?comic=1218) has a punchline with 47 (nonspace) characters. The predicted number of search steps was 1000, the average token probability was 12.6%, "[+]" means the search succeeded, taking 589 steps (0.6 * prediction), covering a probability pass of 94.404%, discarding a probability mass of 0.098% worth of low-probability branches, leaving 5.498% unexplored. The process took 23 seconds.

In `tok_scores/<mod>.txt", you'll find a breakdown of the punchline:

```
T-Rex (punchline): You
                  know        what           ?          It           '           s      really       weird        when         you         say        that           .
prb             7.464%     35.875%     17.110%      3.208%     72.217%     97.738%      1.036%      1.450%      2.592%     28.401%     19.554%     31.999%     27.813%
rln scr        60.502%     68.755%     78.337%     57.337%     80.066%     78.918%     93.231%     89.725%     79.063%     77.428%     74.551%    100.000%    100.000%
pfx scr        59.103%    156.420%     48.877%      2.946%      4.783%      7.525%      2.097%      0.811%      0.763%      5.604%     47.852%   4943.276%   6401.743%
tok ahd              1           0           1           5           0           0          18          11           8           1           0           0           0
val ahd              1           0           1           1           0           0           4           4           4           0           0           0           0
prb ahd         41.32%       0.00%      43.58%      57.21%       0.00%       0.00%      65.62%      63.88%      77.71%      32.36%       0.00%       0.00%       0.00%
min score: 0.7630%  optimistic cost: 1.00e3  average prob: 12.6%  overall_prob: 2.00e-10%  average tok time: 40175 
```

  * `prb`: Raw probability of the token, from the LLM.
  * `rln scr`: Probability assigned by the Remaining Letters Neural Net to the letters left at this point.
  * `pfx scr`: Score assigned to that prefix
  * `tok ahd`: Number of tokens with a higher probability at this point
  * `val ahd`: Of those tokens, the number which are valid (consistent with the letter pool and other hints).
  * `prb ahd`: Sum of the probabilities of the tokens ahead.

The `val ahd`s are used to set the "optimistic cost", which is the number before the `/` in the cost calibration runs.
