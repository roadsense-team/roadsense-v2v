from queue import PriorityQueue
from dataclasses import dataclass

@dataclass(frozen=True)
class Msg:
    val: int

@dataclass(frozen=True)
class Recv:
    m: Msg

pq = PriorityQueue()
# Add two items with same priority and same secondary key (vehicle_id)
# This forces comparison of the third element (Recv object)
try:
    pq.put((100, "V1", Recv(Msg(1))))
    pq.put((100, "V1", Recv(Msg(2))))
    print("Items added")
    print(pq.get())
    print(pq.get())
except TypeError as e:
    print(f"CRASHED: {e}")
