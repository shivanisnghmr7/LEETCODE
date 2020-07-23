"""
141. https://leetcode.com/problems/linked-list-cycle/

Summary: 
1. Keep 1 slow pointer, keep 1 fast pointer (runs with the double speed). if they meet, then cycle exists.

Time/Space: O(n)/O(1)
"""

def linked-list-cycle(head):
    if head is None: return False
    try:
        slow, fast = curr, fast.next
        while slow != fast:
            slow, fast = slow.next, fast.next.next
        return True
    except:
        return False

"""
203. https://leetcode.com/problems/remove-linked-list-elements/

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
24. https://leetcode.com/problems/swap-nodes-in-pairs/

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