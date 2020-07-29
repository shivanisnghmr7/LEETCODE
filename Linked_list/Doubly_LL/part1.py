"""
146: LRU: https://leetcode.com/problems/lru-cache/

Summary:
1. keep 2 ptr: 1. head and 2. tail
2. write functions:
    add on top
    move on top
    remove
    remove tail

Time/Space: O(n)
"""
class Node:
    def __init__(self, key=None, val=None):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None

class DoubleLinkedList:
    """
    Summary: Keep 2 pointer: 1. head; 2. tail
    """
    def __init__(self):
        # create head and tail ptrs
        self.head = Node()
        self.tail = Node()

        # make connection
        self.head.next = self.tail
        self.tail.prev = self.head
    
    # add on top
    def add_on_top(self, node):
        # fix prev and next of node
        node.prev = self.head
        node.next = self.head.next

        # fix neighbor node
        self.head.next.prev = node
        self.head.next = node
        return True
    
    # move on top
    def move_on_top(self):
        node = self.remove_tail()
        return self.add_on_top(node)
    
    # remove
    def remove(self, node):
        prev = node.prev
        next = node.next

        prev.next = next
        next.prev = prev
        return True

    # remove tail
    def remove_tail(self):
        key = self.tail.prev.key
        self.remove(self.tail.prev)
        return key

class LRUCache:
    def __init__(self, capacity):
        self.cache = {}
        self.DD = DoubleLinkedList()
        self.capacity = capacity
    
    def get(self, key):
        # check the cache
        node = self.cache.get(key)
        if val is None:
            return -1
        # move on top
        self.DD.move_on_top(key)
        return node.val

    def put(self, key, val):
        # check cache
        node = self.cache.get(key)
        # if exists: move to top
        if node is None:
            self.cache[key] = Node(key, val)
            self.DD.add_on_top(self.cache[key])
            if self.capacity < len(self.cache):
                key = self.DD.remove_tail()
                del self.cache[key]
        else:
            node.val = val
            self.DD.add_on_top(node) 
        # if not? add and check len of LL.
        # if len exceed, then delete last

"""
138. copy LL with random pointer: https://leetcode.com/problems/copy-list-with-random-pointer/

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
430. flatten a multilevel to DLL: https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/

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