import re

def rouge_l(prediction, reference):
    a, b = re.findall(r"\w+", prediction.lower()), re.findall(r"\w+", reference.lower())
    if not a or not b:
        return 0.0
    dp = [0]*(len(b)+1)
    for word in a:
        previous = 0
        for j, target in enumerate(b, 1):
            old = dp[j]
            dp[j] = previous+1 if word == target else max(dp[j], dp[j-1])
            previous = old
    return 2*dp[-1]/(len(a)+len(b))
