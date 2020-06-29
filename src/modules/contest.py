# import pdb
# T = input()
# cases = []
# cases.append(input())
# cases = cases[0].split(' ')
# for case in cases:
# 	prev, curr = 0, 1 
# 	for j in range(int(case)-1):
# 		temp = curr
# 		curr+= prev
# 		prev = temp
# 	print(curr)


# def mod(a, m): 
#     return (a%m + m) % m 

# print(mod(100000237, 10))

# import pdb

# import numpy as np

# def argmax(a):
#     return max(range(len(a)), key=lambda x: a[x])

# T = input()
# cases = []
# cases.append(input())
# cases = cases[0].split(' ')
# cases = list(map(lambda x: int(x), cases))
# iters = input()
# for j in range(int(T)):
#     su = 0
#     for i in range(int(iters)):
#         max_n, ind = max(cases), argmax(cases)
#         cases[ind] = cases[ind] - 1
#         su+= max_n
# print(su)

import pdb
T = input()
cases = []
cases.append(input())
cases = cases[0].split(' ')
cases = list(map(lambda x: int(x), cases))

dp = [0, 1, 1]
for k in range(int(T)):
    dp = [0]*int(cases[k])
    if len(dp) >1:
        case = cases[k]
        dp[1],  dp[2]= 1, 1
        for j in range(2, case):
            dp[j] = dp[j-1] + dp[j-2]
    print(sum(dp)+1)