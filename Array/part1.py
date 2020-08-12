"""
Questions covered:
One pointer:
    2 Sum: https://leetcode.com/problems/two-sum/
    Contains Duplicate: https://leetcode.com/problems/contains-duplicate/
    Best Time to Buy and Sell Stocks: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    Best Time to Buy and Sell Stocks ii : https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
    Best Time to Buy and Sell Stocks iii : https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
    Jump game: https://leetcode.com/problems/jump-game/
    Minimum steps in Jump game: https://leetcode.com/problems/jump-game-ii/
    count prime number: https://leetcode.com/problems/count-primes/
    Maximum Subarray: https://leetcode.com/problems/maximum-subarray/
    Monotonic array: https://leetcode.com/problems/monotonic-array/

Two pointer:
    2 Sum ii: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
    Product of Array Except Self: https://leetcode.com/problems/product-of-array-except-self/
    remove duplicates in sorted array: https://leetcode.com/problems/remove-duplicates-from-sorted-array/
    remove duplicates more than 2 times in sorted array: https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
    next permutation: https://leetcode.com/problems/next-permutation/
    longest palindromic substring: https://leetcode.com/problems/longest-palindromic-substring/
    Container With Most Water: https://leetcode.com/problems/container-with-most-water/
    Trapping Rain Water: https://leetcode.com/problems/trapping-rain-water/

Three Pointer:
    Dutch national flag: https://leetcode.com/problems/sort-colors/
    3 Sum: https://leetcode.com/problems/3sum
    3 Sum Closest: https://leetcode.com/problems/3sum-closest/
    Merge sorted array with no space: https://leetcode.com/problems/merge-sorted-array/
    Make Maximum number from string: https://leetcode.com/problems/maximum-swap/

Quick sort:
    Kth largest element in array: https://leetcode.com/problems/kth-largest-element-in-an-array/

Sliding the window pattern:
    Subarray sum equal to K: https://leetcode.com/problems/subarray-sum-equals-k/

Merge Interval:
    Merge Intervals: https://leetcode.com/problems/merge-intervals/
    Interval list intersection: https://leetcode.com/problems/interval-list-intersections/

Subsets: 
    find subset of array: https://leetcode.com/problems/subsets/
    find subset of arary with duplicates: https://leetcode.com/problems/subsets-ii/
    permutation: https://leetcode.com/problems/permutations/
    Generate Parentheses: https://leetcode.com/problems/generate-parentheses/

Math:   
    plus one: https://leetcode.com/problems/plus-one/
    multiple 2 string: https://leetcode.com/problems/multiply-strings/

Matrix:
    Leftmost col with at least a one: https://leetcode.com/problems/leftmost-column-with-at-least-a-one/

API:
    Shuffle array: https://leetcode.com/problems/shuffle-an-array/
    random pick index: https://leetcode.com/problems/random-pick-index/

Matrix:
    Leftmost col with at least a one: https://leetcode.com/problems/leftmost-column-with-at-least-a-one/
"""

"""
***************************************

            1 POINTER

***************************************
"""

"""
1. 2 Sum: https://leetcode.com/problems/two-sum/

Summary:
1. dict = {}
2. if num exists in dict.
    return dict[num], i
3. else:
    dict[target-num] = i
return -1 (target not found)

Time/Space: O(n)/O(n)
"""

def two_sum(arr):
    dict = {}
    for i, c in enumerate(arr):
        if dict.get(c) != None:
            return [dict[c], i]
        dict[target-c] = i
    return -1

"""
217. Contains Duplicate: https://leetcode.com/problems/contains-duplicate/

Summary:
set = ()
keep checking in set before adding

Time/Space: O(n)/O(n)
"""

def contains_duplicates(nums):
    s = set()
    for num in nums:
        if num in s: return True
        s.add(num)
    return False

"""
121. Best Time to Buy and Sell Stocks: https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

Summary:
1. profit_today; max_profit= 0; minium= float('inf')
2. find profit_today = profit_here-minium
3. update minium
4. update max_profit

Time/Space: O(n)/O(1)
"""

def Best_time_stock_i(prices):
    max_profit= 0; minium= float('inf')
    for profit_here in prices:
        #profit_today = profit_here-minium
        minium = min(minium, profit_here)
        max_profit = max(max_profit, profit_here-minium)
    return max_profit

"""
122. Best Time to Buy and Sell Stocks ii : https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/

Summary:
1. testcase: if len == 1
maximum_profit = 0
if prices[i-1]<prices[i]:
    maximum_profit+= prices[i]-prices[i-1]

Time/Space: O(n)/O(1)
"""

def Best_time_stock_ii(prices):
    if len(prices) == 1: return 0
        maximum_profit = 0
        for i in range(1, len(prices)):
            if prices[i-1] < prices[i]:
                maximum_profit+= prices[i]-prices[i-1]
        return maximum_profit

"""
123. Best Time to Buy and Sell Stocks atmost 2 time : https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/

Summary:
1. find the 1st max_profit
2. find the 2nd max_profit ( exclude 1st transaction profit from 2nd transaction)
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
5. longest palindromic substring: https://leetcode.com/problems/longest-palindromic-substring/

Summary:
1. keep 2 ptr.

Time/Space: O(n^2)/O(1)
"""

def LPS(arr):
    n = len(res)
    if n < 2: return n
    res = ""
    for i in range(n):
        tmp = helper(s, i, i)
        if len(tmp) > len(res):
            res = tmp
        tmp = helper(s, i, i+1)
        if len(tmp) > len(res):
            res = tmp
    return res

def helper(s, i, j):
    while i >= 0 and j < len(s) and s[i] == s[j]:
        i -= 1
        j += 1
    return s[i+1:j]

"""
647. Count Palindromic Substring: https://leetcode.com/problems/palindromic-substrings/

Summary:
1. total number of palindromic will be 2n-1
2. find center, left from cente and right from center
3. start comparing left and right for palindrome

Time/Space: O(n2)/O(1)
"""

def count_Palindrome(arr):
    n = len(arr)
    count = 0
    for center in range(2*n-1):
        left = center//2
        right = left + (center%2)
        while left >= 0 and right < n and arr[left] == arr[right]:
            count += 1
            left -= 1
            right += 1
    return count

"""
125. valid palindrome: https://leetcode.com/problems/valid-palindrome/

Summary:
1. keep 2 ptr: 1. left and right
2. if found space or not alphabet then incr
3. if alpha is found:
    compare alpha lower

Time/Space: O(n)/O(1)
"""

def valid(arr):
    left, right = 0, len(arr)-1
    while left < right:
        if not arr[left].isalpha(): left += 1
        if not arr[right].isalpha(): right -= 1
        else:
            if arr[left].lower() != arr[right].lower():
                return False
            left += 1
            right -= 1
    return True

"""
680. delete one character to make palindrome: https://leetcode.com/problems/valid-palindrome-ii/

Summary:
1. keep 2 ptr.
2. if not equal:
try skipping one character

Time/Space: O(n)/O(1)
"""

def delete_palindrome(arr):
    left, right = 0, len(arr)-1
    while left < right:
        if arr[left] != arr[right]:
            one, two = arr[left:right], arr[left+1: right+1]
            return one == one[::-1] or two == two[::-1]
        left += 1
        right -= 1
    return True

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
53. Maximum Subarray: https://leetcode.com/problems/maximum-subarray/

Summary:
# testcase: if nothing in arr
imax = sum = nums[0]
if adding this next number is going to inc the sum.
    add next num in sum
else
    update sum with curr
update imax

Time/Space: O(n)/O(1)
"""

def max_subarray(nums):
    if not nums: return 0
        imax = sum = nums[0]
        for i in range(1, len(nums)):
            if sum+nums[i] > nums[i]:
                sum += nums[i]
            else:
                sum = nums[i]
            imax = max(imax, sum)
        return imax

"""
152. Maximum Product Subarray: https://leetcode.com/problems/maximum-product-subarray/

Summary:
1. small = big = maxim = nums[0]; skip the first element
2. keep one product of big num, and one of small num  (bcz of negative)
    to check if the product will change big or small
3. find the maximum of big and small

Time/Space: O(n)/O(1)
"""

def mps(nums):
    small = big = maxim = nums[0]
    for i in nums[1:]:
        small, big = min(i, i*big, i*small), max(i, i*big, i*small) # to check if the product will change big or small
        maxim = max(maxim, small, big)
    return maxim

"""
896. Monotonic array: https://leetcode.com/problems/monotonic-array/

Summary:
keep inc and dec flag as True
update flag

Time/Space: O(n)/O(1)
"""

def Monotonic(A):
    inc= dec = True
    for i in range(1, len(A)):
        if A[i-1] < A[i]:
            dec = False
        elif A[i-1] > A[i]:
            inc = False
    return inc or dec

"""
***************************************

            2 POINTER

***************************************
"""

"""
167. 2 Sum ii in sorted array: https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/

Summary:
1. left = 0; right = len(arr)-1
if arr[left] == arr[right]:
    return [left+1, right+!]
if arr[left]+arr[right] > target:
    reduce right
else
    inc left

Time/Space: O(n)/O(1)
"""

def two_Sum_ii(arr, target):
    left = 0; right = len(arr)-1
    while left < right:
        if arr[left] == arr[right]:
            return [left+1, right+!]
        if arr[left]+arr[right] > target:
            right -= 1
        else:
            left += 1
    return

"""
26. remove duplicates in sorted array: https://leetcode.com/problems/remove-duplicates-from-sorted-array/

Summary:
1. keep boundary= p; count= 1; iterator
2. inc count if repeating
3. if count <= 2: overwrite with boundry and inc boundary
4. return len

Time/Space: O(n)/O(1)
"""

def remove_dupliactes(arr):
    p = count= 1
        for i in range(1, len(nums)):
            if nums[i-1] == nums[i]: count += 1
            else: count = 1
            
            if count < 2:
                nums[p] = nums[i]
                p += 1
        return p

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
238. Product of Array Except Self: https://leetcode.com/problems/product-of-array-except-self/

Summary:
1. keep a dp of same length as nums and make dp[0] = 1
2. multi the prev dp and num. place it in curr position.
3. keep var which keep cumulative product from behind (curr+1). then product the var with dp curr. place it in curr

Time/Space: O(n)/O(n)
"""

def PAES(nums):
    # create dp
    dp = [0 for _ in range(len(nums))]
    # fill first position
    dp[0] = 1
    # 1st round (product of prev dp and prev num)
    for i in range(1, len(dp)):
        dp[i] = dp[i-1]*nums[i-1]
        
    # 2nd round with cumulative var. product the var with curr position and place it
    var = 1
    for i in range(len(dp)-2, -1, -1):
        var = nums[i+1]*var
        dp[i] = var*dp[i]
    # return dp
    return dp

"""
11. Container With Most Water: https://leetcode.com/problems/container-with-most-water/

Summary:
1. keep 2 ptr: L and R; max_area = 0
2. capture max_area
3. if heigh L < heigh R: inc L else dec R

Time/Space: O(n)/O(1)
"""

def CWMW(height):
    left = imax = 0
    right = len(height)-1
    while left < right:
        imax = max(imax, (right-left)*min(height[left], height[right]))
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return imax

"""
42. Trapping Rain Water : https://leetcode.com/problems/trapping-rain-water/

Summary:
# keep 2 ptr: L, R
maintain tallest height of left and right
if lh < rh:
    add lh to vol - height of building
    inc Left
else:
    add rh to vol - height of building
    inc right
return vol

Time/Space: O(n)/O(1)
"""

def Trapping_Rain_water(height):
    L, R = 0, len(height)-1
    lh = rh = vol = 0
    while L < R:
        lh, rh = max(lh, height[L]), max(rh, height[R])
        if lh < rh:
            vol += lh - height[L]; L += 1
        else:
            vol += rh - height[R]; R -= 1
    return vol

"""
***************************************

            3 POINTER

***************************************
"""

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
15. 3 Sum: https://leetcode.com/problems/3sum/

Summary:
sort()
3 ptr: i, L=i+1(L runs +1 from i); R
if i doesn't have duplicates
    find sum = nums[i]+nums[L]+nums[R]
    if sum < target: inc L
    elif sum > target: dec R
    else: # if on target
        append
        remove duplicates by inc L and dec R
        inc L and dec R (for duplicates)
return res

Time/Space: O(n2)/O(1)
"""

def threeSum(nums):
    res = []
    nums.sort()
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]: continue
        L, R = i+1, len(nums)-1
        while L< R:
            sum = nums[i]+nums[L]+nums[R]
            if sum > 0: R -= 1
            elif sum < 0: L += 1
            else:
                res.append([nums[i], nums[L], nums[R]])
                # to remove duplicates
                while L < R and nums[L] == nums[L+1]:
                    L += 1
                while L < R and nums[R] == nums[R-1]:
                    R -= 1
                L += 1; R -= 1
    return res

"""
16. 3 Sum Closest: https://leetcode.com/problems/3sum-closest/

Summary:
keep 1 diff
keep 3 ptr: i, L=i+1, R (exactly like 3 Sum without duplicates)
addition: if target-sum < diff then update diff
testcase: if diff == 0: break 
return target-diff (return sum)
"""

def 3SumClosest(nums, target):
    diff = float('inf')
    for i in range(len(nums)):
        L, R = i+1, len(nums)-1
        while L<R:
            sum = nums[i]+nums[L]+nums[R]
            #update diff
            if abs(target-sum) < abs(diff):
                diff = target-sum
            if sum > target: R-=1
            else: L+= 1
        if diff == 0: break
    return target-diff

"""
88. Merge sorted array with no space: https://leetcode.com/problems/merge-sorted-array/

Summary:
start from behind
# keep 3 ptr. len(nums1), len(nums2), len(nums1+nums2)
# keep added the greater one in the end
# if still array is left, append it

Time: O(n+m)/O(1)
"""

def merge(nums1, m, nums2, n):
    p1, p2 = m-1, n-1
    p = m+n-1
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums1[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    # leftout
    nums1[:p2+1] = nums2[:p2+1]
    return nums1

"""
670. Find maximum number: https://leetcode.com/problems/maximum-swap/

Summary:
create str to array
keep 3 ptr: maximum number index, x, y
start reading the array from end
    if element > maximum number index then replace maximum number index with elem
    if maximum number index > element: update x with curr_index and y with max number index
In the end, we will have max and min (x and y)
swap x with y
create arr to str

Time/Space: O(n)/O(1)
"""

def maximum_swap(num):
    arr = [int(x) for x in num]
    max_index, x, y = len(arr)-1, 0, 0
    for i in range(len(arr)-1, -1, -1):
        if arr[max_index] < arr[i]:
            max_index = i
        elif arr[max_index] > arr[i]:
            x, y = i, max_index
    arr[x], arr[y] = arr[y], arr[x]
    return int(''.join(str(x) for x in arr))

"""
215. Kth largest element in array: https://leetcode.com/problems/kth-largest-element-in-an-array/

Summary: Implement Quick Sort
1. find pivot
    a. make high as pivot and keep low
    b. if found element < pivot
        swap with the low and inc low boundary
    c. swap pivot and high
    d. return pivot
2. do quick sort
    a. find pivot
    b. if len(arr)-k > pivot 
        find in right most half # because pivot is center
    c. if len(arr)-k < pivot 
        find in left most half
    d. else return pivot elem

Time/Space: O(nlogn)/O(1)
"""

def Quick_Sort(arr):
    return quicksort(arr)

def quicksort(arr):
    pivot, n = find_pivot(arr), len(arr)
    if n-k > pivot:
        return quicksort(arr[pivot+1:])
    if n-k < pivot:
        return quicksort(arr[:pivot])
    else:
        return arr[pivot]

def find_pivot(arr):
    low, high = 0, len(arr)
    pivot = arr[high]
    for i in range(low, high):
        if arr[i] < pivot:
            arr[i], arr[low] = arr[low], arr[i]
            low += 1
    arr[high], arr[low] = arr[low], arr[high]
    return low

"""
***************************************

            SLIDING THE WINDOW

***************************************
"""

"""
560.	Subarray sum equal to K: https://leetcode.com/problems/subarray-sum-equals-k/

Summary:
1. maintain sum cumulatively bcz we have to find continuous, dict and count (total continuous subarray sum)
2. whenever sum-k exists then inc count with dict[sum-k]
3. also, add sum in dict and keep inc it if exists.

Time: O(n)/O(n)
"""

def SSK(nums, k):
    sum = count = 0
    dict = {0:1}
    for num in nums:
        # cumulative sum
        sum += int(num)
        # if sum-k exits
        if sum-k in dict: count += dict[sum-k]
        # keep every level sum in dict
        if dict.get(sum) is None: dict[sum] = 0
        dict[sum] += 1
    return count

"""
***************************************

        MERGE INTERVAL

***************************************
"""

"""
56. Merge Intervals: https://leetcode.com/problems/merge-intervals/

Summary:
sort()
if 2nd element of prev >= 1st element of curr:
    find a, b
    replace a, b with curr
    pop the prev
else: inc curr
return arr

Time/Space: O(n)/O(1)
"""

def merge_intervals(intervals):
    i= 1
    intervals.sort()
    while i < len(intervals):
        print(i,intervals[i-1][1], intervals[i][1] )
        if intervals[i-1][1] >= intervals[i][0]:
            a, b = min(intervals[i-1][0], intervals[i][0]), max(intervals[i-1][1], intervals[i][1])
            intervals[i] = [a, b]
            intervals.pop(i-1)
        else:
            i+=1
    return intervals

"""
986. Interval list intersection: https://leetcode.com/problems/interval-list-intersections/

Summary:
keep 2 ptr for A, B length
find max(1st_A, 1st_B) and min(2nd_A, 2nd_B)
whoever has less b value; inc that pointer

Time: O(n+m)/O(1)
"""

def interval_list(A, B):
    p=q=0
    res = []
    while p < len(A) and q < len(B):
        a, b = max(A[p][0], B[q][0]), min(A[p][1], B[q][1])
        if a <= b: res.append([a, b])
        if B[q][1] < A[p][1]:
            q+= 1
        else:
            p += 1
    return res

"""
***************************************

            SUBSET

***************************************
"""

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
***************************************

            DFS

***************************************
"""

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
***************************************

            ADDITION

***************************************
"""

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
***************************************

            API

***************************************
"""

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

"""
***************************************

            MATRIXS

***************************************
"""

"""
1428. Leftmost col with at least a one: https://leetcode.com/problems/leftmost-column-with-at-least-a-one/

Summary:
find dimensions of matrix
take 2 ptr: r, c
start reading c from end
if matrix is 0 == inc r
else dec col
return col+1
# testcase: if no 1

Time/Space: O(n+m)/O(1)
"""

def leftmost(binaryMatrix):
    row, col = binaryMatrix.dimensions()
    r, c = 0, col-1
    while r < row and c >= 0:
        if binaryMatrix.get(r, c) == 0:
            r+=1
        else:
            c-=1
    
    if c == col-1:
        return -1
    return c+1

"""
766. Diagonal comparison : https://leetcode.com/problems/toeplitz-matrix/

Summary:
1. iterate like normal matrix and compare with next diagonal
Note: Dont read the last ele bcz they dont have any diagonal

Time/Space:O(n)/O(1)
"""

def diagonal(matrix):
    return all(matrix[r][c] == matrix[r+1][c+1] for r in range(len(matrix)-1) for c in range(len(matrix[0]-1)))