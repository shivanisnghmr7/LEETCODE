
"""
138. https://leetcode.com/problems/copy-list-with-random-pointer/

Summary:
1. create A->A'->B->B'->C->C' from A->B->C (exactly like insert)
2. map random pointers (A.next.random = A.random.next )
3. seperate 2 LL's

Time: O(n)/O(1)
"""

def copy_list_with_random_ptr(head):
    if head is None: return None

    # Step1
    curr = head
    while curr:
        new = Node(curr.val)
        # standard insert
        next = curr.next
        new.next = next
        curr.next = new
        curr = next
    
    # Step2
    curr = head
    while curr:
        curr.next.random = curr.random.next if curr.random else None

    # Step3
    old = head
    copy_new = new = head.next
    while old:
        old.next = old.next.next
        new.next = new.next.next if new.next else None
        old = old.next
        new = new.next
    return copy_new

"""
430. https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/

Summary:
1. keep 2 pointers: 1. prev, 2. dummy
2. keep stack and add head in it
3. if stack has elem.
    a. pop()
    a1. make node DLL
    b. find next/child and put in stack. also remove next/child from DLL
    c. update prev
4. fix dummy links to SLL
"""

def flatten_a_multilevel_doubly_LL(head):
    if not head: return
    stack, dummy = [head], dummy(0, None, head, None) #val, prev, next, child
    prev = dummy
    while stack:
        node = stack.pop()
        # create link for node
        node.prev = prev
        prev.next = node
        if node.next:
            stack.append(node.next)
            node.next = None
        if node.child:
            stack.append(node.child)
            node.child = None
        prev = node
    # make dummy singly
    dummy.prev = None
    return dummy.next