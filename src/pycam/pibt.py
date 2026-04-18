"""
a toy PIBT implementation taken from
https://github.com/Kei18/pypibt
"""

import numpy as np

from .dist_table import DistTable
from .mapf_utils import Config, Coord, get_neighbors, Action, calculate_action

MAX_OCCUPANCY: int = 2
"""Value indicating maximum number of agents allowed to occupy a location."""

PARALLEL_ACTIONS: list[Action] = [(0, 0, 1), (0, 0, -1)]  # d_z, d_y, d_x 
"""Actions that are in parallel to the x-axis"""


class PIBT:
    def __init__(self, dist_tables: list[DistTable], goals: Config, seed: int = 0) -> None:
        self.N = len(dist_tables)
        assert self.N > 0
        self.dist_tables = dist_tables
        self.grid = self.dist_tables[0].grid
        self.goals = goals

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full((*self.grid.shape, MAX_OCCUPANCY), self.NIL, dtype=int)
        self.occupied_nxt = np.full((*self.grid.shape, MAX_OCCUPANCY), self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int, merging_actions: list[Action]) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize
        C = sorted(C, key=lambda u: self.dist_tables[i].get(u))

        # vertex assignment
        for v in C:
            # # avoid vertex collision
            # if self.occupied_nxt[v] != self.NIL:
            #     continue            

            # # avoid edge collision
            # if j != self.NIL and Q_to[j] == Q_from[i]:
            #     continue

            # check vertex collision
            if any(agent != self.NIL for agent in self.occupied_nxt[v]) and (
                self.goals[i] != v
            ):
                continue
            # check stacking of agents
            if any(agent != self.NIL for agent in self.occupied_nxt[(int(not v[0]), *v[1:])]):
                continue
            # check edge collision (diagonals)
            action: Action = calculate_action(v, Q_from[i])
            if sum(abs(val) for val in action) == 2:
                if abs(action[0]):  #y-diag
                    crossing_nodes = [(int(not Q_from[i][0]), *Q_from[i][1:]), (int(not v[0]), *v[1:])]
                else:
                    crossing_nodes = [(0, Q_from[i][1], v[2]), (0, v[1], Q_from[i][2])]
                v_j_from = set(self.occupied_now[crossing_nodes[0]])
                v_j_to = set(self.occupied_nxt[crossing_nodes[1]])

                if any(agent != self.NIL for agent in v_j_from) and any(agent != self.NIL for agent in v_j_to):
                    common = v_j_from.intersection(v_j_to)
                    common.discard(self.NIL)                    
                    if len(common) > 0:
                        continue
            # check merging in parallel
            other_agent = [agent for agent in self.occupied_nxt[v] if agent != self.NIL]
            if other_agent:
                #assert len(other_agent) == 1? just as sanity check
                v_j_from = Q_from[other_agent[0]]
                other_action: Action = calculate_action(v, v_j_from)
                if action not in PARALLEL_ACTIONS or (v_j_from != v and other_action not in PARALLEL_ACTIONS):
                    continue
            # check splitting in parallel
            if all(agent != self.NIL for agent in self.occupied_now[Q_from[i]]):    # 2 agents in v_i_from
                tos: list[Coord] = [Q_to[agent] for agent in self.occupied_now[Q_from[i]] if Q_to[agent] != self.NIL_COORD]
                if len(tos) == 2 and tos[0] != tos[1]:  # if both goals are determined and they are splitting
                    # check the action for the one(s) moving is in parallel
                    actions: list[Action] = [calculate_action(Q_to[agent], Q_from[agent]) for agent in self.occupied_now[Q_from[i]]]
                    if any(action != (0, 0, 0) and action not in PARALLEL_ACTIONS for action in actions):
                        continue
                    # check the action for the one(s) moving is in opposite direction
                    if merging_actions:
                        merging_actions: list[Action] = [merging_actions[agent] for agent in self.occupied_now[Q_from[i]]]
                        if any(action != (0, 0, 0) and action != (-merging_actions[k][0], -merging_actions[k][1], -merging_actions[k][2]) for k, action in enumerate(actions)):
                            continue


            # reserve next location
            Q_to[i] = v
            idx = np.argmax(self.occupied_nxt[v] == self.NIL)
            self.occupied_nxt[v][idx] = i

            # priority inheritance (j != i checked originally with no edge collision, so added manually here)
            # j = self.occupied_now[v]
            # if (
            #     # j != i
            #     all(agent != i for agent in j) 
            #     and j != self.NIL
            #     and (Q_to[j] == self.NIL_COORD)
            #     and (not self.funcPIBT(Q_from, Q_to, j))
            # ):
            #     continue
            agents = self.occupied_now[v]
            if (
                any(
                    j != i 
                    and j != self.NIL 
                    and (Q_to[j] == self.NIL_COORD) 
                    and (not self.funcPIBT(Q_from, Q_to, j, merging_actions)) 
                    for j in agents
                )                 
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def step(
        self,
        Q_from: Config,
        Q_to: Config,
        order: list[int],
        merging_actions: list[Action],
    ) -> bool:
        flg_success = True

        # setup
        for i, (v_i_from, v_i_to) in enumerate(zip(Q_from, Q_to)):
            idx = np.argmax(self.occupied_now[v_i_from] == self.NIL)
            self.occupied_now[v_i_from][idx] = i
            if v_i_to != self.NIL_COORD:
                # #  check vertex collision
                # if self.occupied_nxt[v_i_to] != self.NIL:
                #     flg_success = False
                #     break
                # # check edge collision
                # j = self.occupied_now[v_i_to]
                # if j != self.NIL and j != i and Q_to[j] == v_i_from:
                #     flg_success = False
                #     break
                # self.occupied_nxt[v_i_to] = i

                # check vertex collision
                if any(agent != self.NIL for agent in self.occupied_nxt[v_i_to]) and (
                    self.goals[i] != v_i_to
                ):
                    flg_success = False
                    break
                # check stacking of agents
                if any(agent != self.NIL for agent in self.occupied_nxt[(int(not v_i_to[0]), *v_i_to[1:])]):
                    flg_success = False
                    break
                # check edge collision (diagonals)
                action: Action = calculate_action(v_i_to, v_i_from)
                if sum(abs(val) for val in action) == 2:
                    if abs(action[0]):  #y-diag
                        crossing_nodes = [(int(not v_i_from[0]), *v_i_from[1:]), (int(not v_i_to[0]), *v_i_to[1:])]
                    else:
                        crossing_nodes = [(0, v_i_from[1], v_i_to[2]), (0, v_i_to[1], v_i_from[2])]
                    v_j_from = set(self.occupied_now[crossing_nodes[0]])
                    v_j_to = set(self.occupied_nxt[crossing_nodes[1]])

                    if any(agent != self.NIL for agent in v_j_from) and any(agent != self.NIL for agent in v_j_to):
                        common = v_j_from.intersection(v_j_to)
                        common.discard(self.NIL)                    
                        if len(common) > 0:
                            flg_success = False
                            break
                # check merging in parallel
                other_agent = [agent for agent in self.occupied_nxt[v_i_to] if agent != self.NIL]
                if other_agent:
                    #assert len(other_agent) == 0? just as sanity check
                    v_j_from = Q_from[other_agent[0]]
                    other_action: Action = calculate_action(v_i_to, v_j_from)
                    if action not in PARALLEL_ACTIONS or (v_j_from != v_i_to and other_action not in PARALLEL_ACTIONS):
                        flg_success = False
                        break
                # check splitting in parallel
                if all(agent != self.NIL for agent in self.occupied_now[v_i_from]):    # 2 agents in v_i_from
                    tos: list[Coord] = [Q_to[agent] for agent in self.occupied_now[v_i_from] if Q_to[agent] != self.NIL_COORD]
                    if len(tos) == 2 and tos[0] != tos[1]:  # if both goals are determined and they are splitting
                        # check the action for the one(s) moving is in parallel
                        actions: list[Action] = [calculate_action(Q_to[agent], Q_from[agent]) for agent in self.occupied_now[v_i_from]]
                        if any(action != (0, 0, 0) and action not in PARALLEL_ACTIONS for action in actions):
                            flg_success = False
                            break
                        # check the action for the one(s) moving is in opposite direction
                        if merging_actions:
                            merging_actions: list[Action] = [merging_actions[agent] for agent in self.occupied_now[v_i_from]]
                            if any(action != (0, 0, 0) and action != (-merging_actions[k][0], -merging_actions[k][1], -merging_actions[k][2]) for k, action in enumerate(actions)):
                                flg_success = False
                                break

                idx = np.argmax(self.occupied_nxt[v_i_to] == self.NIL)
                self.occupied_nxt[v_i_to][idx] = i

        # perform PIBT
        if flg_success:
            for i in order:
                if Q_to[i] == self.NIL_COORD:
                    flg_success = self.funcPIBT(Q_from, Q_to, i, merging_actions)
                    if not flg_success:
                        break

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            if q_to != self.NIL_COORD:
                self.occupied_nxt[q_to] = self.NIL

        return flg_success
