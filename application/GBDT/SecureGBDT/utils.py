import os

import numpy as np

def get_subtree_leaves(tree, index):
    """返回给定节点的左子树和右子树的所有叶子节点索引"""
    left_leaves = []
    right_leaves = []

    left_child = 2 * index + 1
    right_child = 2 * index + 2

    # 查找左子树叶子节点
    if left_child < len(tree):
        left_leaves = bfs_for_leaves(tree, left_child)

    # 查找右子树叶子节点
    if right_child < len(tree):
        right_leaves = bfs_for_leaves(tree, right_child)

    return left_leaves, right_leaves


def bfs_for_leaves(tree, start_index):
    """广度优先搜索子树中的叶子节点"""
    leaves = []
    queue = [start_index]

    while queue:
        current = queue.pop(0)
        left = 2 * current + 1
        right = 2 * current + 2

        # 如果当前节点没有子节点，则是叶子节点
        if left >= len(tree) and right >= len(tree):
            leaves.append(current)
        else:
            if left < len(tree):
                queue.append(left)
            if right < len(tree):
                queue.append(right)

    return leaves
def get_subtree_leaves(tree, index):
    """返回给定节点的左子树和右子树的所有叶子节点索引"""
    left_leaves = []
    right_leaves = []

    left_child = 2 * index + 1
    right_child = 2 * index + 2

    # 查找左子树叶子节点
    if left_child < len(tree):
        left_leaves = bfs_for_leaves(tree, left_child)

    # 查找右子树叶子节点
    if right_child < len(tree):
        right_leaves = bfs_for_leaves(tree, right_child)

    return left_leaves, right_leaves


def bfs_for_leaves(tree, start_index):
    """广度优先搜索子树中的叶子节点"""
    leaves = []
    queue = [start_index]

    while queue:
        current = queue.pop(0)
        left = 2 * current + 1
        right = 2 * current + 2

        # 如果当前节点没有子节点，则是叶子节点
        if left >= len(tree) and right >= len(tree):
            leaves.append(current)
        else:
            if left < len(tree):
                queue.append(left)
            if right < len(tree):
                queue.append(right)

    return leaves

class Tree:
    def __init__(self, f=-1, t=-1):
        self.f = f
        self.t = t

    def __str__(self):
        return f"f={self.f}, t={self.t}"





def path(x, tree_list, tree_indices, p, h):
    start_index = 0
    delta = 2 ** (h - 1) - 1
    queue = [start_index]
    while queue:
        current_index = queue.pop(0)
        left = 2 * current_index + 1
        right = 2 * current_index + 2
        if tree_list[current_index].m == -1:
            if left < 2 ** (h - 1) - 1:
                queue.append(left)
            if right < 2 ** (h - 1) - 1:
                queue.append(right)
        else:
            left_index, right_index = get_subtree_leaves(tree_indices, current_index)
            l = np.array(left_index) - delta
            r = np.array(right_index) - delta

            if x[tree_list[current_index].m]<tree_list[current_index].t:
                p[l.tolist()] = 1
                p[r.tolist()] = 0
                if left < 2 ** (h - 1) - 1:
                    queue.append(left)
            else:
                p[r.tolist()] = 1
                p[l.tolist()] = 0
                if right < 2 ** (h - 1) - 1:
                    queue.append(right)
    return p

