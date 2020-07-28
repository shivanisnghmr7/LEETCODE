"""
203. remove element from linked list (if duplicates) https://leetcode.com/problems/remove-linked-list-elements/

Summary:
1. Questions to ask: if the val exists multiple times?
2. create a dummy.
3. keep dummy as prev and head as curr.
4. if the val is found on curr.val then remove the node.

Time/Space: O(n)/O(1)
"""

def remove-linked-list-elements(head, val):
    if head is None: return None

    # create dummy
    dummy = TreeNode(0)
    dummy.next = head

    prev, curr = dummy, curr
    while curr:
        if curr.val == val:
            prev.next = curr.next
        else:
            prev = curr
        curr = curr.next
    return head

"""
206: https://leetcode.com/problems/reverse-linked-list/

Summary:
1. keep 3 pointer:
    a. prev
    b. curr
    c. next
2. Reverse:
   next --> curr.next
   curr.next --> prev
3. stage:
    prev = curr
    curr = next
4. move head to end

Time/Space: O(n)/O(1)
"""

def reverse-linked-list(head):
    if head is None: return None
    curr, prev = head, None
    while curr:
        # reverse
        next = curr.next
        curr.next = prev
        # stage
        prev = curr
        curr = next
    # handle head
    head = curr
    return head

"""
141. find cycle in Linked List: https://leetcode.com/problems/linked-list-cycle/

Summary: 
1. Keep 1 slow pointer, keep 1 fast pointer (runs with the double speed). if they meet, then cycle exists.

Time/Space: O(n)/O(1)
"""

def linked_list_cycle(head):
    if head is None: return False
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow, fast = slow.next, fast.next.next
        if slow == fast: return slow
    return False

"""
142. find the start of Linked List cycle: https://leetcode.com/problems/linked-list-cycle-ii/

Summary:
1. use slow and fast pointer to detect loop.
2. after loop is detected, 
    a. keep 1 ptr at head and another at the intersection point.
    b. keep moving with normal speed.
    c. if the meet at single point, that is intersection

Time/Space: O(n)/O(1)
"""
def linked_list_cycle_ii(head):
    if head is None: return None
    # find cycle refer up
    joint = linked_list_cycle(head)
    if joint is False: return False
    # find loop point
    ptr1, ptr2 = head, joint
    while ptr1 != ptr2:
        ptr1, ptr2 = ptr1.next, ptr2.next
    return ptr1

"""
160. intersection of Linked List: https://leetcode.com/problems/intersection-of-two-linked-lists/

Summary:
1. read list1 and list2, if one length is smaller than another.
example: A= [1, 3, 4, 5, 9, 10]; B = [1, 4, 5, 9]
2. after 1 of list reading is complete, then make ptrB start from head of A.
    and vice versa.
3. difference of both A and B will reach intersection

Time/Space: O(n)/O(1)
"""

def intersection_of_two_linked_lists(headA, headB):
    if headA is None or headB is None: return None
    ptrA, ptrB = headA, headB
    while ptrA != ptrB:
        ptrA = headB if ptrA is None else ptrA = ptrA.next
        ptrB = headA if ptrB is None else ptrB = ptrB.next
    return ptrA

"""
21. merge 2 sorted Linked List: https://leetcode.com/problems/merge-two-sorted-lists/

Summary:
1. create dummy, curr.
whatever is smaller value, add that value after dummy.
2. test case: if one list gets over.

Time/Space: O(n)/O(1)
"""

def merge_two_sorted_lists(l1, l2):
    dummy = curr = Node(0)
    while l1 and l2:
        if l1.val > l2.val:
            dummy.next = l2
            l2 = l2.next
        else:
            dummy.next = l1
            l1 = l1.next
        curr = curr.next
    if l1 or l2:
        curr.next = l1 or l2
    return dummy.next

"""
19. remove nth node from end of the Linked List: https://leetcode.com/problems/remove-nth-node-from-end-of-list/

Summary:
1. testcase: if len of linked list < n
2. move fast ahead by n
3. then move normal and remove slow
"""

def remove_nth_node_from_end_of_list(head, n):
    if head is None: return None
    slow = fast = head
    for _ in range(n):
        fast = fast.next
    # testcase: if len of linked list < n
    if fast is None: return head.next
    while fast.next:
        slow, fast = slow.next, fast.next
    slow.next = slow.next.next
    return head

"""
61. rotate Linked List: https://leetcode.com/problems/rotate-list/

Summary:
1. find the rotating index (specially if k > len of LL)
2. rotate it using fast and slow pointer like (find the nth element from last)
3. there are 4 important test case: 
#   1. head is None
#   2. list have only 1 element
#   3. if k == length of LL
#   4. if k is 0

Time/Space: O(n)/O(1)
"""

def rotate_list(head, k):
    # testcase 1:
    if head is None: return None
    # testcase 2:
    if head.next is None: return head

    # find k, L
    k, L = find_length(head, k)
    # testcase 3:
    if k == L: return head
    # testcase 4:
    if k == 0: return head

    slow = fast = head
    for i in range(k):
        fast = fast.next

    while fast.next:
        slow, fast = slow.next, fast.next.next
    h2 = slow.next
    slow.next = None
    fast.next = h2
    head = h2
    return head

def find_length(head, k):
    curr, L = head, 0
    while curr:
        L += 1
        curr = curr.next
    if k > L: return k%L, L
    return k, L

"""
143. zig zag in linked list: https://leetcode.com/problems/reorder-list/

Summary:
1. find the mid using fast and slow pointer
2. reverse after mid elements
3. merge 2 list in alternate fashion 
"""

def reorder_list(head):
    if head is None: return None

    # find mid
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    
    # reverse the half after mid
    prev, curr = None, slow
    while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = curr.next
    
    # merge 2 list
    first, second = head, prev
    while second.next:
        first.next, first = second, first.next
        second.next, second = first, second.next
    return head

"""
24. swap nodes in pairs: https://leetcode.com/problems/swap-nodes-in-pairs/

Summary:
1. keep 3 pointers:
    a. prev
    b. first
    c. second
2. create dummy and make prev.
3. iterate: if head and head.next is True
    a. mark nodes to be swapped (first and second)
    b. swap prev --> second --> first --> second.next
    c. stage prev(first) and head(second) for next swap

Time/Space: O(n)/O(1)
"""

def swap_nodes_in_pairs(head):
    if head is None: return None
     # create dummy
     dummy = TreeNode(0)
     dummy.next = head
     # mark prev
     prev = dummy
     # iterate
     while (head and head.next): # because swapping in pair
        # mark swap nodes
        first = head;
        second = head.next

        # swaping
        prev.next = second
        first.next = second.next
        second.next = first

        # stage for swap
        prev = first
        head = first.next
    return dummy.next

"""
25. reverse nodes in k group: https://leetcode.com/problems/reverse-nodes-in-k-group/

Summary:
1. make fast k times faster than slow
2. reverse k times
3. extend the dummy to head of reverse
4. adjust for next round

Edge case: if len < k: return dummy.next

Time/Space: O(n)/O(1)
"""

def reverse_k_group(head):
    dummy = ptr = Node(0)
    dummy.next = slow = fast = head

    while True:
        count = 0
        while fast and count < k:
            fast = fast.next
            count += 1
        if count == k:
            prev, curr = fast, slow
            for _ in range(k):
                next = curr.next
                curr.next = prev
                prev = curr
                curr = next
            ptr.next = prev
            ptr = slow
            slow = fast
        else:
            return dummy.next

"""
2. add two number: https://leetcode.com/problems/add-two-numbers/

Summary:
1. keep 2 ptr: a. dummy and b. curr
2. find if l1 or l2 or carry
3. find if l1.val, l2.val and find carry, divide
 divmod(v1+v2+carry, 10) = carry, val
3. create node(val)

Time/space: O(n)/O(1)
"""

def add_two_number(l1, l2):
    dummy = curr = Node(0)
    carry = 0
    while carry or l1 or l2:
        v1 = v2 = 0
        if l1:
            v1 = l1.val
            l1 = l1.next
        if l2:
            v2 = l2.val
            l2 = l2.next
        carry, val = divmod(v1+v2+carry, 10)
        curr.next = Node(val)
        curr = curr.next
    return dummy.next

"""
445: add two number of different length: https://leetcode.com/problems/add-two-numbers-ii/

Summary:
1. create numbers from linked list
2. add them
3. reconstruct linked list by insert on head
"""

def add_two_numbers_ii(l1, l2):
    num1 = construct_numb(l1)
    num2 = construct_numb(l2)
    num = num1+num2
    head = None
    while num:
        node = Node(num%10)
        if head is None:
            head = node
        else:
            node.next = head
            head = node
        num //= 10

def construct_numb(link):
    num = 0

    while link:
        num *= 10
        num += link.val
        link = link.next
    return num

"""
109. sorted LL to BST: https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/

Summary:
1. find the size of Linked List
2. create a BST recursively inorder (go left, create node, update next parent, go right)

Time/Space: O(n)/O(logn)
"""
def convert_sorted_list_to_binary_tree(head):
    if head is None: return None

    # find size
    size = findsize(head)
    # BST inorder recursive
    def BST_inorder(left, right):
        nonlocal head
        # invalid for BST
        if left > right: return None
        mid = (left+right)//2
        # process left
        left = BST_inorder(left, mid-1)
        # node
        node = TreeNode(head)
        # update new parent
        head = head.next
        # assign left
        node.left = left
        # assign right
        node.right = BST_inorder( mid+1, right)
    return BST_inorder(0, size-1)

def findsize(head):
    count, curr = 0, head
    while curr:
        curr = curr.next
        count += 1
    return count