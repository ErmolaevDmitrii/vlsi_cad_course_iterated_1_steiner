#!/usr/bin/env python3
from __future__ import annotations
from enum import Enum, unique
from typing import Optional, Tuple, ClassVar, List, overload
from random import choice, uniform, randint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import time
import heapq
import bisect
import argparse
import os
import sys


class Tree:
    __last_id: ClassVar[Optional[int]] = None

    @staticmethod
    def __generate_id() -> int:
        if Tree.__last_id is None:
            Tree.__last_id = -1
        Tree.__last_id += 1
        return Tree.__last_id

    @staticmethod
    def check_tree_id(id: int) -> bool:
        if Tree.__last_id is not None:
            return id <= Tree.__last_id
        return False

    @property
    def id(self) -> int:
        return self.__id

    def __init__(self, vertexes: Optional[List[Tree.Vertex]]):
        self.__id: int = Tree.__generate_id()
        self.__vertex_id_set: set[int] = set()
        self.__max_vertex_id: int      = -1

        if vertexes is not None:
            for vertex in vertexes:
                vertex.attach_vertex_to_tree(self)

        self.__edge_set: Tree.EdgeSet = Tree.EdgeSet(self, vertexes)

    def __generate_vertex_id(self) -> int:
        self.__max_vertex_id += 1
        self.__vertex_id_set.add(self.__max_vertex_id)
        return self.__max_vertex_id

    def __add_vertex_id(self, id: int) -> None:
        assert self.__check_vertex_id(id), f"вершина с id={id} есть в дереве"
        self.__max_vertex_id = id if id > self.__max_vertex_id else self.__max_vertex_id
        self.__vertex_id_set.add(id)

    def __check_vertex_id(self, id: int) -> bool:
        return id not in self.__vertex_id_set

    def __remove_vertex_id(self, id: int) -> None:
        assert not self.__check_vertex_id(id), f"вершины с id={id} нету в дереве"
        self.__max_vertex_id = id - 1 if id == self.__max_vertex_id else self.__max_vertex_id
        self.__vertex_id_set.remove(id)

    class Vertex:
        @unique
        class Type(Enum):
            TERMINAL = 1
            STEINER  = 2

        def __init__(self, coords: Tuple[int, int], id: Optional[int] = None, type: Optional[Tree.Vertex.Type] = None, tree: Optional[Tree] = None):
            self.__coords: Tuple[int, int]  = tuple(coords)
            self.__type:   Tree.Vertex.Type = type if type is not None else Tree.Vertex.Type.TERMINAL
            self.__id:     Optional[int]    = id
            self.__tree:   Optional[Tree]   = tree

            if self.__tree is not None:
                id = self.__tree.attach_vertex(self)
                if id is not None:
                    self.__id = id

        @property
        def coords(self) -> Tuple[int, int]:
            return tuple(self.__coords)

        @property
        def id(self) -> Optional[int]:
            return self.__id

        @property
        def type(self) -> Tree.Vertex.Type:
            return self.__type

        @property
        def tree_id(self) -> Optional[int]:
            return None if self.__tree is None else self.__tree.id

        @staticmethod
        def distance(vertex_1: Tree.Vertex, vertex_2: Tree.Vertex) -> int:
            return abs(vertex_1.coords[0] - vertex_2.coords[0]) + abs(vertex_1.coords[1] - vertex_2.coords[1])

        def attach_vertex_to_tree(self, tree: Tree) -> None:
            assert self.__tree is None, f"вершина уже является вершиной в другом дереве, tree_id={self.tree_id}"

            self.__tree = tree
            id = tree.attach_vertex(self)
            if id is not None:
                self.__id = id

        def detach_vertex_from_tree(self) -> None:
            assert self.__tree is not None, f"вершина не принадлежит никакому дереву"
            self.__tree.detach_vertex(self)

            self.__id = None
            self.__tree = None

        def __eq__(self, other):
            if other is None:
                return False
            assert isinstance(other, Tree.Vertex), f"other не типа вершины, type(other)={str(type(other))}"
            return self.coords == other.coords

        def __hash__(self):
            return hash(self.coords)

    class EdgeSet:
        def __init__(self, tree: Tree, vertexes: Optional[List[Tree.Vertex]]):
            self.__tree: Tree = tree

            self.__max_vertex_count: int = 1 if vertexes is None else 2 * len(vertexes)
            self.__vertex_count:     int = 0
            self.__graph_weight:     int = 0

            # -1 означает "ребра нет"
            self.__edge_matrix:   np.ndarray = np.full((self.__max_vertex_count, self.__max_vertex_count), -1, dtype=int)
            self.__vertexes:      np.ndarray = np.empty(self.__max_vertex_count, dtype=object)
            self.__vertex_degree: np.ndarray = np.empty(self.__max_vertex_count, dtype=int)
            self.__vertexes.fill(None)

            if vertexes is not None:
                self.__vertex_count = len(vertexes)
                self.__vertexes[:self.vertex_count] = vertexes

            self.__temp_vertex:        Optional[Tree.Vertex] = None
            self.__temp_vertex_edges:  np.ndarray            = np.full(self.__max_vertex_count, -1, dtype=int)
            self.__temp_vertex_degree: int                   = 0

            self.__vertexes_list: List[Tree.Vertex] = []

        @property
        def vertex_count(self) -> int:
            return self.__vertex_count

        @property
        def graph_weight(self) -> int:
            return self.__graph_weight

        @property
        def vertexes(self) -> Tuple[Tree.Vertex, ...]:
            if len(self.__vertexes_list) != self.vertex_count:
                self.__vertexes_list = self.__vertexes.tolist()[:self.vertex_count]
            return tuple(self.__vertexes_list)

        def __expand(self) -> None:
            n: int = self.__max_vertex_count
            self.__max_vertex_count *= 2
            self.__edge_matrix       = np.pad(self.__edge_matrix,       ((0, n), (0, n)), constant_values=-1)
            self.__temp_vertex_edges = np.pad(self.__temp_vertex_edges,  (0, n),          constant_values=-1)
            self.__vertexes          = np.pad(self.__vertexes,           (0, n),          constant_values=None)
            self.__vertex_degree     = np.pad(self.__vertex_degree,      (0, n),          constant_values=0)

        def push_vertex(self, vertex: Tree.Vertex) -> None:
            assert vertex.tree_id == self.__tree.id, f"вершина не принадлежит дереву"
            assert vertex != self.__temp_vertex, f"недопустимо, чтобы временная вершина и новая постоянная вершина совпадали"

            if (self.__max_vertex_count == self.vertex_count):
                self.__expand()
            else:
                self.__edge_matrix      [:, self.vertex_count] = -1
                self.__edge_matrix      [self.vertex_count, :] = -1
                self.__temp_vertex_edges[self.vertex_count]    = -1
                self.__vertex_degree    [self.vertex_count]    = 0

            self.__vertexes[self.vertex_count] = vertex
            self.__vertex_count += 1

        def push_temporary_vertex(self, vertex: Tree.Vertex) -> None:
            if self.__temp_vertex is not None:
                self.pop_temporary_vertex()
            self.__temp_vertex        = vertex
            self.__temp_vertex_degree = 0

        def pop_temporary_vertex(self) -> None:
            assert self.__temp_vertex is not None, f"временно добавленной вершины нету в массиве"

            for i, w in enumerate(self.__temp_vertex_edges):
                if i == self.vertex_count:
                    break

                if w != -1:
                    self.__graph_weight        -= w
                    self.__vertex_degree[i]    -= 1
                    self.__temp_vertex_edges[i] = -1

            self.__temp_vertex = None

        def __check_vertex_ii(self, vertex_ii: int) -> None:
            assert vertex_ii >= 0, f"индекс вершины vertex_ii={vertex_ii} отрицательный"
            assert vertex_ii < self.vertex_count, f"индекс вершины vertex_ii={vertex_ii} выходит за границы, vertex_count={self.vertex_count}"
            return

        def pop_vertex(self) -> Tree.Vertex:
            assert self.vertex_count > 0, f"вершин в дереве нету"

            for i, w in enumerate(self.__edge_matrix[self.vertex_count - 1]):
                if i == self.vertex_count:
                    break

                if w != -1:
                    self.__graph_weight     -= w
                    self.__vertex_degree[i] -= 1

            if self.__temp_vertex is not None:
                w: int = self.__temp_vertex_edges[self.vertex_count - 1]
                if w != -1:
                    self.__graph_weight       -= w
                    self.__temp_vertex_degree -= 1

            self.__vertex_count -= 1

            popped_vertex: Tree.Vertex = self.__vertexes[self.vertex_count]
            return popped_vertex

        @overload
        def add_edge(self, vertex_ii: int) -> None: ...
        @overload
        def add_edge(self, vertex_1_ii: int, vertex_2_ii: int) -> None: ...

        def add_edge(self, vertex_1_ii: int, vertex_2_ii: Optional[int] = None) -> None:
            if vertex_2_ii is None:
                self.__check_vertex_ii(vertex_1_ii)
                assert self.__temp_vertex is not None, f"временной вершины в графе нету, не к чему подсоединять вершину"

                distance: int = Tree.Vertex.distance(self.__vertexes[vertex_1_ii], self.__temp_vertex)
                assert self.__temp_vertex_edges[vertex_1_ii] == -1, \
                    f"ребро между vertex_ii={vertex_1_ii} и временной вершиной уже есть, distance={distance}"

                self.__temp_vertex_edges[vertex_1_ii]  = distance
                self.__graph_weight                   += distance
                self.__vertex_degree[vertex_1_ii]     += 1
                self.__temp_vertex_degree             += 1
            else:
                self.__check_vertex_ii(vertex_1_ii)
                self.__check_vertex_ii(vertex_2_ii)

                distance: int = Tree.Vertex.distance(self.__vertexes[vertex_1_ii], self.__vertexes[vertex_2_ii])
                assert self.__edge_matrix[vertex_1_ii][vertex_2_ii] == -1, \
                    f"ребро между vertex_1_ii={vertex_1_ii} и vertex_2_ii={vertex_2_ii} уже есть, distance={distance}"

                self.__edge_matrix[vertex_1_ii][vertex_2_ii]  = distance
                self.__edge_matrix[vertex_2_ii][vertex_1_ii]  = distance
                self.__graph_weight                          += distance
                self.__vertex_degree[vertex_1_ii]            += 1
                self.__vertex_degree[vertex_2_ii]            += 1

        @overload
        def remove_edge(self, vertex_ii: int) -> None: ...
        @overload
        def remove_edge(self, vertex_1_ii: int, vertex_2_ii: int) -> None: ...

        def remove_edge(self, vertex_1_ii: int, vertex_2_ii: Optional[int] = None) -> None:
            if vertex_2_ii is None:
                self.__check_vertex_ii(vertex_1_ii)
                assert self.__temp_vertex is not None, f"временной вершины в графе нету, не от кого убирать ребро"

                distance: int = self.__temp_vertex_edges[vertex_1_ii]
                assert distance != -1, f"ребра между vertex_ii={vertex_1_ii} и временной вершиной нету"

                self.__temp_vertex_edges[vertex_1_ii]  = -1
                self.__graph_weight                   -= distance
                self.__vertex_degree[vertex_1_ii]     -= 1
                self.__temp_vertex_degree             -= 1
            else:
                self.__check_vertex_ii(vertex_1_ii)
                self.__check_vertex_ii(vertex_2_ii)

                distance: int = self.__edge_matrix[vertex_1_ii][vertex_2_ii]
                assert distance != -1, f"ребра между vertex_1_ii={vertex_1_ii} и vertex_2_ii={vertex_2_ii} нету"

                self.__edge_matrix[vertex_1_ii][vertex_2_ii]  = -1
                self.__edge_matrix[vertex_2_ii][vertex_1_ii]  = -1
                self.__graph_weight                          -= distance
                self.__vertex_degree[vertex_1_ii]            -= 1
                self.__vertex_degree[vertex_2_ii]            -= 1

        def clear_edges(self) -> None:
            self.__graph_weight = 0

            self.__edge_matrix        = np.full((self.__max_vertex_count, self.__max_vertex_count), -1, dtype=int)
            self.__temp_vertex_edges  = np.full(self.__max_vertex_count, -1, dtype=int)
            self.__vertex_degree      = np.zeros(self.__max_vertex_count, dtype=int)
            self.__temp_vertex_degree = 0

        @overload
        def get_distance(self, vertex_ii: int) -> int: ...
        @overload
        def get_distance(self, vertex_1_ii: int, vertex_2_ii: int) -> int: ...

        def get_distance(self, vertex_1_ii: int, vertex_2_ii: Optional[int] = None) -> int:
            if vertex_2_ii is None:
                self.__check_vertex_ii(vertex_1_ii)
                assert self.__temp_vertex is not None, f"временной вершины в графе нету, не от кого брать расстояние"
                distance: int = Tree.Vertex.distance(self.__vertexes[vertex_1_ii], self.__temp_vertex)
                return distance
            else:
                self.__check_vertex_ii(vertex_1_ii)
                self.__check_vertex_ii(vertex_2_ii)
                distance: int = Tree.Vertex.distance(self.__vertexes[vertex_1_ii], self.__vertexes[vertex_2_ii])
                return distance

        @overload
        def get_edge(self, vertex_ii: int) -> Optional[int]: ...
        @overload
        def get_edge(self, vertex_1_ii: int, vertex_2_ii: int) -> Optional[int]: ...

        def get_edge(self, vertex_1_ii: int, vertex_2_ii: Optional[int] = None) -> Optional[int]:
            if vertex_2_ii is None:
                self.__check_vertex_ii(vertex_1_ii)
                assert self.__temp_vertex is not None, f"временной вершины в графе нету, не от кого брать расстояние"
                edge_length: int = self.__temp_vertex_edges[vertex_1_ii]
                return edge_length if edge_length != -1 else None
            else:
                self.__check_vertex_ii(vertex_1_ii)
                self.__check_vertex_ii(vertex_2_ii)
                edge_length: int = self.__edge_matrix[vertex_1_ii][vertex_2_ii]
                return edge_length if edge_length != -1 else None

        @overload
        def get_vertex_degree(self) -> int: ...
        @overload
        def get_vertex_degree(self, vertex_ii: int) -> int: ...

        def get_vertex_degree(self, vertex_ii: Optional[int] = None) -> int:
            if vertex_ii is None:
                assert self.__temp_vertex is not None, f"временной вершины в графе нету, не от кого брать степень"
                return self.__temp_vertex_degree
            else:
                self.__check_vertex_ii(vertex_ii)
                return self.__vertex_degree[vertex_ii]

        @property
        def edges_list(self) -> List[Tuple[int, int, int]]:
            edges_list: List[Tuple[int, int, int]] = []

            for i in range(self.vertex_count):
                for j in range(i):
                    w = self.get_edge(i, j)
                    if w is not None:
                        edges_list.append((i, j, w))

            if self.__temp_vertex is not None:
                for i in range(self.vertex_count):
                    w = self.get_edge(i)
                    if w is not None:
                        edges_list.append((self.vertex_count, i, w))

            return edges_list

    def attach_vertex(self, vertex: Tree.Vertex) -> Optional[int]:
        if vertex.id is None:
            id: int = self.__generate_vertex_id()
            return id
        else:
            assert self.__check_vertex_id(vertex.id), f"вершина с id={vertex.id} уже существует в дереве"
            self.__add_vertex_id(vertex.id)
            return None

    def detach_vertex(self, vertex: Tree.Vertex) -> None:
        assert vertex.tree_id == self.id, f"вершина не принадлежит дереву с id={self.id}, id дерева вершины = {vertex.tree_id}"
        self.__remove_vertex_id(vertex.id)

    def push_vertex(self, vertex: Tree.Vertex):
        vertex.attach_vertex_to_tree(self)
        self.__edge_set.push_vertex(vertex)

    def pop_vertex(self):
        popped_vertex: Tree.Vertex = self.__edge_set.pop_vertex()
        popped_vertex.detach_vertex_from_tree()

    def push_temporary_vertex(self, vertex: Tree.Vertex):
        self.__edge_set.push_temporary_vertex(vertex)

    def pop_temporary_vertex(self):
        self.__edge_set.pop_temporary_vertex()

    def add_edge(self, vertex_1_ii: int, vertex_2_ii: int) -> None:
        self.__edge_set.add_edge(vertex_1_ii, vertex_2_ii)

    def remove_edge(self, vertex_1_ii: int, vertex_2_ii: int) -> None:
        self.__edge_set.remove_edge(vertex_1_ii, vertex_2_ii)

    def get_distance(self, vertex_1_ii: int, vertex_2_ii: int) -> int:
        return self.__edge_set.get_distance(vertex_1_ii, vertex_2_ii)

    def get_edge(self, vertex_1_ii: int, vertex_2_ii: int) -> Optional[int]:
        return self.__edge_set.get_edge(vertex_1_ii, vertex_2_ii)

    def add_temp_edge(self, vertex_ii: int) -> None:
        self.__edge_set.add_edge(vertex_ii)

    def remove_temp_edge(self, vertex_ii: int) -> None:
        self.__edge_set.remove_edge(vertex_ii)

    def get_temp_distance(self, vertex_ii: int) -> int:
        return self.__edge_set.get_distance(vertex_ii)

    def get_temp_edge(self, vertex_ii: int) -> Optional[int]:
        return self.__edge_set.get_edge(vertex_ii)

    def clear_edges(self) -> None:
        self.__edge_set.clear_edges()

    def get_vertex_degree(self, vertex_ii: int) -> int:
        return self.__edge_set.get_vertex_degree(vertex_ii)

    def get_temp_vertex_degree(self) -> int:
        return self.__edge_set.get_vertex_degree()

    @property
    def tree_weight(self) -> int:
        return self.__edge_set.graph_weight

    @property
    def vertexes(self) -> Tuple[Tree.Vertex, ...]:
        return self.__edge_set.vertexes

    @property
    def edges_list(self) -> List[Tuple[int, int, int]]:
        return self.__edge_set.edges_list

    def __len__(self):
        return self.__edge_set.vertex_count

    def visualise(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        vertexes = self.__edge_set.vertexes
        coords = np.array([(v.coords[0], v.coords[1]) for v in vertexes])

        for i, vertex in enumerate(vertexes):
            ax.scatter(vertex.coords[0], vertex.coords[1], s=30,
                       color='purple' if i == 0 else 'blue' if vertex.type == Tree.Vertex.Type.TERMINAL else 'red', zorder=2)

        for i in range(len(vertexes)):
            for j in range(i):
                if self.__edge_set.get_edge(i, j) is not None:
                    x1, y1 = coords[i]
                    x2, y2 = coords[j]

                    path_type = choice(['hv', 'vh'])

                    if path_type == 'hv':
                        mid = (x1, y2 + uniform(-0.05, 0.05))
                    else:
                        mid = (x2 + uniform(-0.1, 0.1), y1)

                    ax.plot([x1, mid[0]], [y1, mid[1]], color='gray', linewidth=1.5, zorder=1)
                    ax.plot([mid[0], x2], [mid[1], y2], color='gray', linewidth=1.5, zorder=1)

        return fig, ax

def load_vertexes(filename: str) -> List[Tree.Vertex]:

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data["node"]

    vertexes: List[Tree.Vertex] = []

    for node in nodes:
        node_id = node['id']
        node_x, node_y = node['x'], node['y']
        assert node['type'] in ['t', 's'], f"тип вершины должен быть =['t', 's'], type={node['type']}"
        node_type = Tree.Vertex.Type.TERMINAL if node['type'] == 't' else Tree.Vertex.Type.STEINER
        vertexes.append(Tree.Vertex(coords=(node_x, node_y), id=node_id, type=node_type))

    return vertexes

def build_MST(tree: Tree) -> None:
    tree.clear_edges()
    INF: int = 10**23

    vertex_count: int = len(tree)

    key:    List[int]           = [INF]   * vertex_count
    p:      List[Optional[int]] = [None]  * vertex_count
    in_mst: List[bool]          = [False] * vertex_count

    key[0] = 0

    for _ in range(vertex_count):
        u = -1
        min_key = INF

        for i in range(vertex_count):
            if not in_mst[i] and key[i] < min_key:
                min_key = key[i]
                u = i

        assert u != -1, f"граф несвязный"

        in_mst[u] = True

        for v in range(vertex_count):
            weight = tree.get_distance(u, v)
            if v == u:
                continue
            if not in_mst[v] and weight < key[v]:
                key[v] = weight
                p[v] = u


    for v in range(1, vertex_count):
        if p[v] is not None:
            tree.add_edge(p[v], v)

def iterated_1_steiner(tree: Tree):
    build_MST(tree)

    candidates: List[Tree.Vertex] = generate_steiner_candidates(tree)

    while True:
        min_weight = tree.tree_weight

        best_candidate: Optional[Tree.Vertex] = None

        for i, candidate in enumerate(candidates):
            tree.push_vertex(candidate)
            build_MST(tree)
            new_tree_len = tree.tree_weight

            if new_tree_len < min_weight:
                best_candidate = Tree.Vertex(coords=candidate.coords, type=Tree.Vertex.Type.STEINER)
                min_weight = new_tree_len

            tree.pop_vertex()

        if best_candidate is not None:
            tree.push_vertex(best_candidate)
            candidates.remove(best_candidate)
            build_MST(tree)
        else:
            break

    build_MST(tree)

def generate_steiner_candidates(tree: Tree) -> List[Tree.Vertex]:
    vertexes: List[Tree.Vertex] = tree.vertexes

    existing_coords: List[(int, int)] = {v.coords for v in vertexes}

    xs: List[int] = sorted({v.coords[0] for v in vertexes})
    ys: List[int] = sorted({v.coords[1] for v in vertexes})

    candidates = []

    for x in xs:
        for y in ys:
            if (x, y) not in existing_coords:
                candidates.append(Tree.Vertex(coords=(x,y), type=Tree.Vertex.Type.STEINER))

    return candidates

def add_temp_vertex_to_mst(tree: Tree, vertex: Tree.Vertex, base_edges_list: List[Tuple[int, int, int]], candidate_dists: List[Tuple[int, int]]) -> int:
    tree.push_temporary_vertex(vertex)

    vertex_count: int = len(tree)

    temp_edges_list: List[Tuple[int, int, int]] = [(vertex_count, i, w) for i, w in candidate_dists]
    parent:          List[int]                  = list(range(vertex_count+1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        root_a = find(a)
        root_b = find(b)

        if root_a == root_b:
            return False

        parent[root_a] = root_b
        return True

    def merged_edges(left, right):
        i = j = 0
        n, m = len(left), len(right)
        while i < n and j < m:
            if left[i][2] <= right[j][2]:
                yield left[i]
                i += 1
            else:
                yield right[j]
                j += 1
        yield from left[i:]
        yield from right[j:]

    edges_count:     int = 0
    temp_mst_weight: int = 0

    for u, v, w in merged_edges(base_edges_list, temp_edges_list):
        if union(u, v):
            temp_mst_weight += w
            edges_count += 1
            if edges_count == vertex_count:
                break

    tree.pop_temporary_vertex()
    return temp_mst_weight

def iterated_1_steiner_optimised(tree: Tree):
    build_MST(tree)

    candidates, candidate_dists, d_mins = generate_steiner_candidates_optimised(tree)

    while True:
        start_weight = tree.tree_weight

        best_candidate:    Optional[Tree.Vertex] = None
        best_candidate_ii: Optional[int]         = None

        best_delta:      int = 0
        max_edge_weight: int = max(tree.edges_list, key=lambda edge: edge[2])[2]

        searched_candidates: int = 0
        base_edges_list: List[Tuple[int, int, int]] = tree.edges_list
        base_edges_list.sort(key=lambda edge: edge[2])

        edge_boxes: List[Tuple[int, int, int, int]] = generate_edge_boxes(tree)

        candidates_mask: List[bool] = generate_candidates_mask(candidates, edge_boxes)

        heap = [(-(max_edge_weight - d_mins[i]), i) for i in range(len(candidates)) if candidates_mask[i]]
        heapq.heapify(heap)

        while heap:
            neg_pot, i = heapq.heappop(heap)
            potential = -neg_pot

            if potential <= best_delta:
                break

            candidate = candidates[i]
            searched_candidates += 1
            new_tree_len = add_temp_vertex_to_mst(tree, candidate, base_edges_list, candidate_dists[i])
            delta = start_weight - new_tree_len

            if delta > best_delta:
                best_candidate = Tree.Vertex(coords=candidate.coords, type=Tree.Vertex.Type.STEINER)
                best_candidate_ii = i
                best_delta = delta

        if best_candidate is None:
            break

        candidates[best_candidate_ii] = None
        d_mins    [best_candidate_ii] = None

        vertex_count: int = len(tree)

        for i, candidate in enumerate(candidates):
            if candidate is None:
                continue

            bisect.insort(candidate_dists[i], (vertex_count, Tree.Vertex.distance(candidate, best_candidate)), key=lambda edge: edge[1])
            d_mins[i] = candidate_dists[i][0][1]

        tree.push_vertex(best_candidate)
        build_MST(tree)

    build_MST(tree)

def generate_candidates_mask(candidates: List[Tree.Vertex], edge_boxes: List[Tuple[int, int, int, int]]) -> List[bool]:
    if len(candidates) == 0 or len(edge_boxes) == 0:
        return [False] * len(candidates)

    valid_indexes: List[int]             = []
    valid_coords:  List[Tuple[int, int]] = []

    for i, candidate in enumerate(candidates):
        if candidate is not None:
            valid_indexes.append(i)
            valid_coords.append(candidate.coords)

    if len(valid_indexes) == 0:
        return [False] * len(candidates)

    np_valid_indexes: np.ndarray = np.array(valid_indexes)
    np_valid_coords:  np.ndarray = np.array(valid_coords)
    np_edge_boxes:    np.ndarray = np.array(edge_boxes)

    x: np.ndarray = np_valid_coords[:, 0]
    y: np.ndarray = np_valid_coords[:, 1]

    inside_x_borders: np.ndarray = (x[:, None] >= np_edge_boxes[None, :, 0]) & (x[:, None] <= np_edge_boxes[None, :, 1])
    inside_y_borders: np.ndarray = (y[:, None] >= np_edge_boxes[None, :, 2]) & (y[:, None] <= np_edge_boxes[None, :, 3])

    inside: np.ndarray = inside_x_borders & inside_y_borders

    hit = np.any(inside, axis=1)

    np_candidates_mask: np.ndarray = np.zeros(len(candidates), dtype=bool)

    np_candidates_mask[np_valid_indexes] = hit

    candidates_mask: List[bool] = np_candidates_mask.tolist()

    return candidates_mask


def generate_edge_boxes(tree: Tree) -> List[Tuple[int, int, int, int]]:
    vertexes: List[Tree.Vertex] = tree.vertexes

    edge_boxes: List[Tuple[int, int, int, int]] = []

    for u_ii, v_ii, _ in tree.edges_list:
        x1, y1 = vertexes[u_ii].coords
        x2, y2 = vertexes[v_ii].coords
        edge_boxes.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)))

    return edge_boxes

def filter_candidate(candidate: Tree.Vertex, edge_boxes: List[Tuple[int, int, int, int]]) -> bool:
    for (xmin, xmax, ymin, ymax) in edge_boxes:
        if xmin <= candidate.coords[0] <= xmax and ymin <= candidate.coords[1] <= ymax:
            return True
    return False

def generate_steiner_candidates_optimised(tree: Tree) -> Tuple[List[Tree.Vertex], List[List[Tuple[int, int]]], List[int]]:
    vertexes: List[Tree.Vertex] = tree.vertexes

    existing_coords: List[(int, int)] = {v.coords for v in vertexes}

    xs: List[int] = sorted({v.coords[0] for v in vertexes})
    ys: List[int] = sorted({v.coords[1] for v in vertexes})

    edge_boxes: List[Tuple[int, int, int, int]] = generate_edge_boxes(tree)

    candidates: List[Tree.Vertex] = []

    for x in xs:
        for y in ys:
            if (x, y) in existing_coords:
                continue
            candidate = Tree.Vertex(coords=(x, y), type=Tree.Vertex.Type.STEINER)
            if filter_candidate(candidate, edge_boxes):
                candidates.append(candidate)

    max_edge_weight: int = max(tree.edges_list, key=lambda edge: edge[2])[2]

    items = []

    for candidate in candidates:
        dists = sorted(
            ((i, Tree.Vertex.distance(candidate, v)) for i, v in enumerate(vertexes)),
            key=lambda edge: edge[1]
        )
        d_min = dists[0][1]
        potential = max_edge_weight - d_min
        items.append((potential, candidate, dists, d_min))

    items.sort(key=lambda x: x[0], reverse=True)

    sorted_candidates = [item[1] for item in items]
    sorted_distances  = [item[2] for item in items]
    sorted_d_mins     = [item[3] for item in items]

    return sorted_candidates, sorted_distances, sorted_d_mins

def reset_graph(tree: Tree):
    first_steiner_index: Optional[int] = None

    for i, vertex in enumerate(tree.vertexes):
        if vertex.type == Tree.Vertex.Type.STEINER:
            first_steiner_index = i
            break

    if first_steiner_index is None:
        return

    for _ in range(first_steiner_index, len(tree)):
        tree.pop_vertex()

    tree.clear_edges()

def check_results(tree: Tree):
    for i, vertex in enumerate(tree.vertexes):
        if vertex.type == Tree.Vertex.Type.STEINER:
            assert tree.get_vertex_degree(i) >= 3, f"точка Штейнера со степенью {tree.get_vertex_degree(i)} <= 3"

    assert len(tree.edges_list) == len(tree) - 1, f"правильное дерево с {len(tree)} вершинами должно иметь {len(tree) - 1} ребро, а в этом дереве {len(tree.edges_list)} ребер"

def save_tree_to_json(tree: Tree, filename: str) -> None:
    vertexes = tree.vertexes

    max_vertex_id = max((v.id for v in vertexes), default=0)
    next_edge_id = max_vertex_id + 1

    vertex_edge_ids = {v.id: [] for v in vertexes}

    edges_output = []
    for i, j, _ in tree.edges_list:
        u_id = vertexes[i].id
        v_id = vertexes[j].id
        eid = next_edge_id
        next_edge_id += 1

        edges_output.append({
            "id": eid,
            "vertices": [u_id, v_id]
        })
        vertex_edge_ids[u_id].append(eid)
        vertex_edge_ids[v_id].append(eid)

    nodes_output = []
    for v in vertexes:
        nodes_output.append({
            "id": v.id,
            "x": v.coords[0],
            "y": v.coords[1],
            "type": "t" if v.type == Tree.Vertex.Type.TERMINAL else "s",
            "edges": vertex_edge_ids[v.id]
        })

    output = {"node": nodes_output, "edge": edges_output}
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description='Построение дерева Штейнера на манхэттенской плоскости'
    )
    parser.add_argument('input_file', help='Входной JSON‑файл с вершинами')
    parser.add_argument('-o', '--output', help='Выходной JSON‑файл (по умолчанию: <input>_out.json)')
    parser.add_argument('-m', '--modified', action='store_true',
                        help='Использовать модифицированный (оптимизированный) алгоритм')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Сохранить визуализацию в <input>_out.png')
    parser.add_argument('--check', action='store_true',
                        help='Выполнить проверку результата после построения')
    args = parser.parse_args()

    vertexes = load_vertexes(args.input_file)

    assert vertexes, f"Вершины не были загружены"

    initial_terminals = sum(1 for v in vertexes if v.type == Tree.Vertex.Type.TERMINAL)
    print(f"Загружены {len(vertexes)} вершин ({initial_terminals} терминалов).")

    tree = Tree(vertexes=vertexes)

    build_MST(tree)
    initial_mst_len = tree.tree_weight
    print(f"Начальный вес MST: {initial_mst_len}")

    start_time = time.time()

    if args.modified:
        iterated_1_steiner_optimised(tree)
        mode = "оптимизированный"
    else:
        iterated_1_steiner(tree)
        mode = "базовый"

    elapsed = time.time() - start_time

    final_len = tree.tree_weight
    steiner_cnt = sum(1 for v in tree.vertexes if v.type == Tree.Vertex.Type.STEINER)

    if args.check:
        check_results(tree)

    print(f"\n{'='*50}")
    print(f"Алгоритм: {mode}")
    print(f"Начальный вес MST: {initial_mst_len}")
    print(f"Вес Steiner Tree: {final_len}")
    if initial_mst_len > 0:
        improvement = initial_mst_len - final_len
        percent = (improvement / initial_mst_len) * 100
        print(f"Улучшение: {improvement} ({percent:.1f}%)")
    print(f"Время вычислений: {elapsed:.4f} s")
    print(f"Вершин: {len(tree)} (Терминалов: {initial_terminals}, Точек Штейнера: {steiner_cnt})")
    print(f"Ребер: {len(tree.edges_list)}")
    print(f"{'='*50}\n")

    out_name = args.output if args.output else os.path.splitext(args.input_file)[0] + '_out.json'
    save_tree_to_json(tree, out_name)
    print(f"Сохранено в {out_name}")

    if args.visualize:
        img_name = os.path.splitext(args.input_file)[0] + '_out.png'
        fig, ax = tree.visualise()
        fig.savefig(img_name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Визуализация сохранена в {img_name}")

if __name__ == '__main__':
    main()