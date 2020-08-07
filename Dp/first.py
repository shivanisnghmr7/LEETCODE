"""

Questions covered:
Longest Subsequence:
    •	Longest Common Subsequence: https://leetcode.com/problems/longest-common-subsequence/
    •	Edit distance: https://leetcode.com/problems/edit-distance/
    •	Longest Increasing Subsequence: https://leetcode.com/problems/longest-increasing-subsequence/

"""

"""
###########################################################
#                                                         #
#                                                         #
#           PATTERN: 0/1 KNAPSACK                   #
#                                                         #
#                                                         #
###########################################################
"""

"""
###########################################################
#                                                         #
#                                                         #
#           PATTERN: UNBOUNDED KNAPSACK                   #
#                                                         #
#                                                         #
###########################################################
"""

"""
322. Rod cutting/minimum Coin change/maximum Robin cutting: https://leetcode.com/problems/coin-change/

Summary:
1. create a dp (amount+1)
2. iterate over coins available/iterate over coin to amount+1
3. dp = min(dp[i], dp[i-coin]+1)

Time/Space: O(n2)/O(n)
"""

def Minimum_coin_change(coins, amount):
    dp = [True for _ in range(amount+1)]
    dp[0] = 0 # to get 0 coin
    for coin in coins:
        for i in range(coin, amount+1):
            dp[i] = min(dp[i], dp[i-coin]+1)
    if dp[-1] == float('-inf'): return -1
    return dp[-1]

"""
###########################################################
#                                                         #
#                                                         #
#           PATTERN: FIBONACCI QUESTIONS                  #
#                                                         #
#                                                         #
###########################################################
"""

"""
509. fibonacci number: https://leetcode.com/problems/fibonacci-number/

Summary:
1. keep a, b to save last 2 sums
2. edge case: N = 0 and 1

Time/Space: O(n)/O(1)
"""

def fibonacci(N):
    a, b = 0, 1
    if N == 0: return a
    if N == 1: return b
    index = 1
    def helper(N):
        nonlocal index, a, b
        if index == N: return b
        a, b = b, a+b
        index += 1
        helper(N)
    return helper(N)

"""
70. climbing stairs: https://leetcode.com/problems/climbing-stairs/

Summary:
1. ways to climb 1 stairs is 1 way. and way to climb 2 stair is also 1 way.
2. then follow fibannoci

Time/Space: O(n)/O(1)
"""

def climbing_stairs(n):
    a, b = 1, 1
    for i in range(1, n):
        a, b = b, a+b
    return b

"""
climbing stairs with 1 step, 2 step, 3 step

Summary:
1. one step, two step, three step = 1, 1, 2
2. then follow fibannoci

Time/Space: O(n)/O(1)
"""

def climbing_stairs_ii(n):
    one, two, three = 1, 1, 2
    for i in range(n):
        one, two, three = two, three, one+two+three
    return three

"""
198. house robber: https://leetcode.com/problems/house-robber/

Summary:
1. either robber will rob last house or second last house.
2. one variable to maintain max of last two

Time/Space: O(n)/O(1)
"""

def house_robber(arr):
    one, two, three = 0, 0, 0
    for i in arr:
        one, two, three = two, three, max(two, three)
    return max(two, three)

"""
213. house robber ii: https://leetcode.com/problems/house-robber-ii/

Summary:
1. either first house is included or last house
2. edge case arr: no house or 1 house

Time/Space: O(n)/O(1)
"""

def house_robber_ii(arr):

    def helper(arr):
        one, two = 0, 0
        for i in arr:
            one, two = two+i, max(one, two)
        return max(one, two)

    # edge case
    if not arr: return 0
    if len(arr) == 1: return arr[0]
    #either first house is included or last house
    return max(helper(arr[1:]), helper(arr[:-1]))

"""
55. Jump Game: https://leetcode.com/problems/jump-game/

Summary:
1. start in reverse from 2nd last
2. check if the jump (i+nums[i]) can cross last
    if it can, update the last and check if we can reach the start

Time/Space: O(n)/O(1)
"""

def jump_game(arr):
    last = len(arr)-1
    for i in range(last-1, -1, -1):
        if i+nums[i] >= last:
            last = i
    return last == 0

"""
45. Minimum jumps in Jump Game: https://leetcode.com/problems/jump-game-ii/

Summary:
1. create 2 pointer
    a. max_positions covered: maintain max(max_positions, i+nums[i])
    b. max_steps: take max_steps to reach end
2. if max_steps is unable to reach i
    then take one more jump
    update max_steps
3. edge case if arr < 2

Time/Space: O(n)/O(1)
"""

def mini_jump_game(arr):
    n = len(arr)
    if n < 2: return 0
    max_step, max_pos, jump = arr[0], arr[0], 1
    for i in range(n):
        if max_step < i:
            jump += 1
            max_step = max_pos
        max_pos = max(max_pos, i+arr[i])
    return jump

"""
###########################################################
#                                                         #
#                                                         #
#           PATTERN: PALINDROMIC SEQUENCE                 #
#                                                         #
#                                                         #
###########################################################
"""

"""
516. longest palindromic subsequence: https://leetcode.com/problems/longest-palindromic-subsequence/

Summary:
remove 2d into 1d
1. keep 1 old row and keep one new row
2. start scanning from behind (i) and (j) from i+1 to n
3. mark diagonal with 1 in new
4. if s[i] == s[j]:
        new[j] = 2+dp[j-1] # 2+diagonal
    else:
        new[j] = max(dp[j], new[j-1]) 

Time/Space: O(n^2)/O(n)
"""

def LPS(s):
    n = len(s)
    # if palindrome is already sorted
    if s == s[::-1]: return len(s)
    dp = [0 for i in range(n)] # old dp
    dp[n-1] = 1 # mark last diagonal

    for i in range(n-1, -1, -1):
        new = dp[:]
        new[i] = 1
        for j in range(i+1, n):
            if s[i] == s[j]:
                new[j] = 2+dp[j-1] # 2+down diagonal
            else:
                new[j] = max(dp[j], new[j-1])
        dp = new
    return dp[-1]
            


"""
###########################################################
#                                                         #
#                                                         #
#           PATTERN: LONGEST COMMON QUESTIONS             #
#                                                         #
#                                                         #
###########################################################
"""

"""
1143. longest common subsequence: https://leetcode.com/problems/longest-common-subsequence/

Brute-Force:
1. use recursion and compare.
2. increase one of the index and then compare

Time/Space: O(2 ^ (nm))/O(2 ^ (nm))

Bottom-up:
1. make table row: text2+1 and col: text1+1
2. if text1 == text2: extend prev diagonally by 1
else: max(prev hill diagonal)

Time/Space: O(nm)/O(nm)
"""

def Brute_force(text1, text2):
    def helper(text1, text2, i1, i2, c):
        if i1 == len(text1) or i2 == len(text2): return count
        if text1[i1] == text2[i2]:
            c = helper(text1, text2, i1+1, i2+1, count+1)
        c1 = helper(text1, text2, i1, i2+1, 0); c2 = helper(text1, text2, i1+1, i2, 0)
        return max(c, max(c1, c2))
    helper(text1, text2, 0, 0, 0)

def Longest_common(text1, text2):
    row, col = len(text2), len(text1)
    if row*col == 0: return 0
    table = [[0]*(row+1) for _ in range(col+1)]

    for r in range(col):
        for c in range(row):
            if text1[r] == text2[c]:
                table[r+1][c+1] = 1+ table[r][c]
            else:
                table[r+1][c+1] = max(table[r][c+1], table[r+1][c])
    return table[-1][-1]

"""
72. Edit distance: https://leetcode.com/problems/edit-distance/

Summary:
1. create a dp table.
2. make 1st row (0 to m) and make 1st col (0 to n)
3. record
    a. left = dp[i-1][j]+1 
    b. right = dp[i][j-1]+1 
    c. left_down = dp[i-1][j-1]
    if words are not equal then leftdown += 1
4. min(left, right, left_down)

Time/Space: O(nm)/O(nm)
"""

def Edit_distance(word1, word2):
    n, m = len(word1), len(word2)
    if n*m == 0: return n+m
    dp = [[0]*(m+1) for _ in range(n+1)]

    # make boundary
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            left = dp[i-1][j]+1
            right = dp[i][j-1]+1
            left_down = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]: left_down += 1
            dp[i][j] = min(left, right, left_down)
    return dp[n][m]

"""
300. Longest Increasing Subsequence: https://leetcode.com/problems/longest-increasing-subsequence/

Summary:
1. create a dp with 1
2. maintain imax of longest
3. testcase if len(0 or 1)
4. compare all prior to i to num[i]; if inc found, incr in dp

Time/Space: O(n^2)/O(n)
"""

def LIS(arr):
    # testcase
    if len(arr) < 2: return len(arr)
    dp = [1 for i in range(len(arr)+1)]; imax = float('-inf')
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i] and dp[i] < dp[j]+1:
                dp[i] += 1
                imax = max(dp[i], imax)
    if imax == float('-inf'): return 1
    return imax

"""
97. Interleaving String: https://leetcode.com/problems/interleaving-string/

Summary:
1. make s2 as row+1 and s1 as col+1 in dp
2. start iterating and comparing " "+s3 with s1 and s2
3. if s2 == s3 then check if prev is True
4. if s1 == s3 then check if up is True
5. if s1==s2== s3 then check if both prev and up is True

# testcase: s2+s1 != s3

Time/Space: O(n^2)/O(n)
"""

def IS(s1, s2, s3):
    # s1='aabcc'; s2 = 'dbbca'; s3 = 'aadbbcbcac'
    n, m, l = len(s1), len(s2), len(s3)
    if n+m != l: return False
    dp= [True for _ in range(m+1)]
    # dp = [T, T, T, T, T, T]
    # update 1st row
    for j in range(1, m+1):
        # if the prev is True and if m == s3
        dp[j] = dp[j-1] and s2[j-1] == s3[j-1]
    for i in range(1, n+1):
        # if up is True and s1 == s3
        dp[0] = dp[0] and  s1[i-1] == s3[i-1]
        for j in range(1, m+1):
            # if up is True and s1 == s3 or prev is True s2 == s3
            dp[j] = (dp[j] and s1[i-1] == s3[i-1+j]) or (dp[j-1] and s2[j-1] == s3[i-1+j])
    return dp[-1]

"""
1048. Longest String Chain: https://leetcode.com/problems/longest-string-chain/

Summary:
1. sort based on len
2. iterate alphabet
3. maintain dict to map the word with prev extension
4. if dict[word] exists:
    a. find the max of (dict[word], prev word+1)
5. else:
    (prev word+1)

Time/Space: O(n)/O(n)
"""

def LSC(words):
    # create dict
    dict = {}
    # sort based on length
    words.sort(key=len)
    # iterate alpha
    for word in words:
        for i in range(len(word)):
            if word not in dict:
                # if word doesn't exist then extend the prev word
                dict[word] = dict.get(word[:i]+word[i+1:], 0)+1
            else:
                # if word exists, find the max of word and extend
                dict[word] = max(dict[word], dict.get(word[:i]+word[i+1:], 0)+1)
    return max(dict.values())

