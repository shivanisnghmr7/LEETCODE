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
###########################################################
#                                                         #
#                                                         #
#           PATTERN: FIBONACCI QUESTIONS                  #
#                                                         #
#                                                         #
###########################################################
"""

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

