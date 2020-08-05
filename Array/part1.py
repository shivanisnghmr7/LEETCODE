"""
Questions covered:
One pointer:
    remove duplicates in sorted array: https://leetcode.com/problems/remove-duplicates-from-sorted-array/
    Best Time to Buy and Sell Stocks: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    Best Time to Buy and Sell Stocks ii : https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
    Best Time to Buy and Sell Stocks iii : https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
    Jump game: https://leetcode.com/problems/jump-game/
    Minimum steps in Jump game: https://leetcode.com/problems/jump-game-ii/
    count prime number: https://leetcode.com/problems/count-primes/
Two pointer:
    remove duplicates more than 2 times in sorted array: https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
    next permutation: https://leetcode.com/problems/next-permutation/
Three Pointer:
    Dutch national flag: https://leetcode.com/problems/sort-colors/
Subsets: 
    find subset of array: https://leetcode.com/problems/subsets/
    find subset of arary with duplicates: https://leetcode.com/problems/subsets-ii/
    permutation: https://leetcode.com/problems/permutations/
    Generate Parentheses: https://leetcode.com/problems/generate-parentheses/
Math:   
    plus one: https://leetcode.com/problems/plus-one/
    multiple 2 string: https://leetcode.com/problems/multiply-strings/
API:
    Shuffle array: https://leetcode.com/problems/shuffle-an-array/
    random pick index: https://leetcode.com/problems/random-pick-index/
"""

"""
26. remove duplicates in sorted array: https://leetcode.com/problems/remove-duplicates-from-sorted-array/

Summary:
1. keep 1 ptr which compares to next pointer, if its same then pop
2. else inc the ptr

Time/Space: O(n)/O(1)
"""

def remove_dupliactes(arr):
    ptr = 0
    while ptr < len(arr)-1:
        if arr[ptr] == arr[ptr+1]: arr.pop(ptr+1)
        else: ptr += 1
    return arr

"""
121. Best Time to Buy and Sell Stocks: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

Summary:
1. find the minimum value and find the maximum difference.
2. return maximum profit

Time/Space: O(n)/O(1)
"""

def Best_time_stock_i(arr):
    if not arr or len(arr) == 1: return 0
    min_val, max_profit = float('float'), 0
    for i in range(len(arr)):
        profit = arr[i] - min_val
        min_val = min(min_val, arr[i])
        max_profit = max(max_profit, profit)
    return max_profit

"""
122. Best Time to Buy and Sell Stocks ii : https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

Summary:
1. if the next element is greater than curr that means profit. add to profit

Time/Space: O(n)/O(1)
"""

def Best_time_stock_ii(arr):
    if not arr or len(arr) == 1: return 0
    profit = 0
    for i in range(len(arr)-1):
        if arr[i] < arr[i+1]:
            profit += arr[i+1] - arr[i]
    return profit

"""
123. Best Time to Buy and Sell Stocks atmost 2 time : https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/

Summary:
1. find the 1st max_profit
2. find the 2nd max_profit just remove the 1st max while finding min
3. return 2nd profit

Time/Space: O(n)/O(1)
"""

def Best_time_stock_iii(arr):
    first_min, first_max_profit, second_min, second_max_profit = float('inf'), 0, float('inf'), 0
    for i in arr:
        # first round
        first_min = min(first_min, i)
        first_max_profit = max(first_max_profit, i-first_min)
        # second round
        second_min = min(second_min, i-first_max_profit)
        second_max_profit = max(second_max_profit, i-second_min )
    return second_max_profit

"""
55. Jump game: https://leetcode.com/problems/jump-game/

Summary:
1. run from second last; keep variable last = len(arr)-1
2. check if i+arr[i] (current+jump length) >= last
    if so, update last = i
3. check if last == 0

Time/Space: O(n)/O(1)
"""

def Jump_game(arr):
    last = len(arr)-1
    # reverse
    for i in range(len(arr)-2, -1, -1):
        if (i+arr[i]) >= last: last = i
    return last == 0

"""
45. Minimum Jump Game: https://leetcode.com/problems/jump-game-ii/

Summary:
1. keep 3 ptr: 
    a. max_position can be reached 
    b. max_steps one can take 
    c. jumps 
2. maintain max_position by keeping max of (curr+jump_length)
3. if max_steps is < i:
    make max_steps = max_position

Time/Space: O(n)/O(1)
"""

def Mini_Jump_game(arr):
    if len(arr) < 2: return 0

    jump = 0
    max_position = max_steps = arr[0]
    for i in range(1, len(arr)):
        if max_steps < i: # if need one more jump to reach i
            jump += 1
            max_position = max_steps # take one more maximum step
        max_position = max(max_position, arr[i]+i)
    return jump

"""
31. Next Permutation: https://leetcode.com/problems/next-permutation/

Summary:
1. find the first decr ele from behind (1st ptr)
2. from that decr element find the immediate larger number towards right
    a. start from the end and find fist element larger than 1st ptr
3. swap
4. reverse the half after 1st ptr
        
testcase: if reaches the start

Time/Space: O(n)/O(1)
"""

def Next_Permutation(arr):
    arr = nums
    ptr2 = ptr1 = len(nums)-1
    while ptr1 > 0 and arr[ptr1] <= arr[ptr1-1]: # first decr element find
        ptr1 -= 1
    
    if ptr1 == 0: # testcase
        arr.reverse()
        return
    
    ptr1 = ptr1-1 
    while ptr2 > ptr1 and arr[ptr2] <= arr[ptr1]: # find the 1st increasing after decr element from behind
        ptr2 -= 1
    arr[ptr1], arr[ptr2] = arr[ptr2], arr[ptr1] # swap
    
    ptr1, ptr2 = ptr1+1, len(arr)-1
    while ptr2 > ptr1: # reverse after decr element
        arr[ptr1], arr[ptr2] = arr[ptr2], arr[ptr1]
        ptr2 -= 1; ptr1 += 1
    return(arr)

"""
204. count number of prime number: https://leetcode.com/problems/count-primes/

Summary:
1. create a list with n
2. mark 0, 1 as prime
NOTE: loop over only half of array
3. if square and gap of i of number exists in list. make it 0 because they are not prime

Time/Space: O(n)/O(n)
"""

def count_prime(n):
    # testcase
    if n < 2: return 0
    arr = [1]*n
    arr[1]= arr[0]= 0 # mark 0, 1 as prime
    for i in range(2, int(n**0.5)+1): # range is divided by 1/2
        # check if the square exists and mark those places
        if arr[i] == 1: arr[i*i:n:i] = [0]* len(arr[i*i:n:i])
    return sum(arr)

"""
80. remove duplicates more than 2 times in sorted array: https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/

Summary:
1. if count is less or equal to 0. keep moving forward
2. keep j to maintain replacement boundary 

Time/Space: O(n)/O(1)
"""

def remove_duplicates_twice(arr):
    count = j = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]: count += 1
        else: count -= 1

        if count <= 2:
            arr[j] = arr[i]
            j += 1
    return arr[:j] # after j it will be unneccessary element

"""
75. Dutch national flag: https://leetcode.com/problems/sort-colors/

Summary:
1. keep 3 ptrs:
    a. maintain boundary of 0 on left; 
    b. maintain boundary of 2 on right side; 
    c. curr -- iterator
2. if find 0; swap with a and curr; then inc curr and a
3. if find 2; swap with b and curr; dec b
4. found 1; inc curr

Time/Space: O(n)/O(1)
"""

def Dutch_national_flag(arr):
    if not arr: return
    low = curr = 0
    high = len(arr)
    while low < high:
        if arr[curr] == 0:
            arr[curr], arr[low] = arr[low], arr[curr] # swap
            low += 1; curr += 1
        elif arr[curr] == 2:
            arr[curr], arr[high] = arr[high], arr[curr] # swap
            high -= 1
        else:
            curr += 1
    return arr

"""
78. Find the subset of array: https://leetcode.com/problems/subsets/

Summary:
make output = [[]]
1. make a copy of len(output)
2. iterate through the copy of length and copy the list inside output
3. insert the num in copy of list and append to output

Time/Space: O(2^n)/O(2^n)
"""

def subset(arr):
    output = [[]]
    for num in arr:
        L = len(output) #make a copy of len(output)
        for i in range(L):
            new = list(output[i]) #copy the list inside output
            new.append(num) #insert the num in copy of list
            output.append(new)
    return output

"""
90. find subset of arary with duplicates: https://leetcode.com/problems/subsets-ii/

Summary:
1. create output = [[]]
2. sort: so that duplicates will be next to each other
3. check if duplicates exists, if they do then dont include the old list
4. create a new and append num.
5. insert into output

Time: O(2^n)/O(2^n)
"""

def subset_with_duplicates(arr):
    output = [[]]
    arr.sort()
    for i in range(len(arr)):
        if i == 0 or arr[i-1] != arr[i]:
            L = len(output)
        for p in range(len(output)-L, len(output)): # skip the old list if duplicates
            new = list(output[i]) #copy the list inside output
            new.append(num) #insert the num in copy of list
            output.append(new)
    return output

"""
46. Permutation: https://leetcode.com/problems/permutations/

Summary:
1. create DFS(nums, path, output)
2. keep removing from nums and keep forming path
3. if no nums is left then copy the path to output

Discuss: https://leetcode.com/problems/permutations/discuss/18296/Simple-Python-solution-(DFS).

Time/Space: O(N!*N)/O(N!*N)
"""

def permutation_of_array(arr):
    output = []
    dfs(arr, [], output)
    return output

def dfs(arr, path, output):
    if not arr:
        output.append(path)
    for i in range(len(arr)):
        dfs(arr[:i]+arr[i+1], path+[arr[i]], output)

"""
22. Generate Parentheses: https://leetcode.com/problems/generate-parentheses/

Summary:
1. base condition: path == 2*n
2. add left first and inc left
3. add right second and inc right

Discuss: https://leetcode.com/problems/generate-parentheses/discuss/10096/4-7-lines-Python

Time/Space: O(N!*N)/O(N!*N)
"""

def Generate_Parentheses(n):
    res = []
    dfs(n, 0, 0, '', res)
    return res

def dfs(n, left, right, path, res):
    if path == 2*n: res.append(path)
    if left < n: dfs(n, left+1, right, path+"(", res)
    if right < left: dfs(n, left, right+1, path+")", res)

"""
66. plus one: https://leetcode.com/problems/plus-one/

Summary:
1. read from the end.
2. if i is 9: make arr[i] = 0
   else; just add 1 and return
3. for case where all are 999: add 1 in front

Time/Space: O(n)/O(1)
"""

def Plus_one(arr):
    for i in range(len(arr)-1, -1, -1):
        if arr[i] == 9:
            arr[i] = 0
        else:
            arr[i] = arr[i]+1
            return arr
    return [1]+arr

"""
43. Multiple 2 strings : https://leetcode.com/problems/multiply-strings/

Summary:
1. create a list len(nums1)+len(num2)
2. reverse both the strings and start to multiple
    NOTE: don't forget to implement "x"
3. testcase: if there are extra 0 in start. resolve them

Time/Space: O(n^2)/O(n)
"""

def Multiple_strings(arr1, arr2):
    ans = [0]*(len(arr1)+len(arr2))
    start_from = len(ans)-1

    for n1 in range(len(arr1)-1, -1, -1):
        cross = start_from # to maintain the cross
        for n2 in range(len(arr2)-1, -1, -1):
            # multiple
            ans[cross] += int(arr[n1])*int(arr[n2])
            # carry handling
            ans[cross-1] += ans[cross]//10
            # update the position
            ans[cross] = ans[cross]%10
        start_from -= 1

    # testcase: to handle 0 in the start
    ptr = 0
    while ptr < len(ans)-1 and ans[ptr] == 0:
        ptr += 1
    return ''.join(map(str, product[pt:]))

"""
384. Shuffle an array: https://leetcode.com/problems/shuffle-an-array/

Summary:
1. make a copy of array
2. if shuffle: find random index and swap it --> O(n)/O(1)
3. reset: copy the array and return --> O(1)/O(n)

Time/Space: O(n)/O(n)
"""

class Shuffle:
    def __init__(self, arr):
        self.original = self.array = arr

    def shuffle(self):
        for i in range(len(self.array)):
            index = random.randrange(i, len(self.array))
            self.array[i], self.array[index] = self.array[index], self.array[i]
        return self.array
    
    def reset(self):
        self.array = list(self.original)
        return self.array

"""
398. Random pick index: https://leetcode.com/problems/random-pick-index/

Summary:
1. make nums global
2. find target apperance and count them
3. apply randint and check if count == rand_index
4. return i

Time/Space: O(n)/O(1)
"""
class Solution:
    
    def __init__(self, nums: List[int]):
        self.nums = nums
        

    def pick(self, target: int) -> int:
        count, res = 0, None
        for i, c in enumerate(self.nums):
            if target == c:
                count += 1
                rand = random.randint(1, count)
                if count == rand: res = i
        return res