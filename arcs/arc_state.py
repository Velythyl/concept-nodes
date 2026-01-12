from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal


@dataclass
class Arc:
    """Represents a directed edge (arc) between two objects.
    
    Arcs can be either forward arcs (direct relationships) or dependency arcs
    (indicating blocked relationships).
    """
    source: str
    target: str
    weight: float
    label: str
    arc_shortname: str
    arc_type: Literal["forward", "dependency"]
    timestamp_hours: float
    day: int
    hour: float
    blocked_by: str | None = None  # Only used for dependency arcs
    
    def to_dict(self) -> dict[str, Any]:
        """Convert Arc to a dictionary for JSON serialization."""
        d = asdict(self)
        # Remove None values for cleaner JSON output
        return {k: v for k, v in d.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Arc":
        """Create an Arc from a dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            weight=data["weight"],
            label=data["label"],
            arc_shortname=data["arc_shortname"],
            arc_type=data["arc_type"],
            timestamp_hours=data["timestamp_hours"],
            day=data["day"],
            hour=data["hour"],
            blocked_by=data.get("blocked_by"),
        )


@dataclass
class ArcState:
    """Tracks the state of the arc generation simulation.

    Forms a linked list where each node represents one timestep.
    The 'prev' pointer links to the previous timestep's state.
    
    Time is tracked via current_day and current_hour as the single source of truth.
    Simulation ends when current_day/current_hour reach max_day/max_hour.
    """

    current_day: int = 1
    current_hour: float = 0.0
    dt_hours: float = 1.0  # Time step in hours
    max_day: int = 2  # Simulation ends when we reach this day
    max_hour: float = 0.0  # And this hour (e.g., day 2, hour 0 = end of day 1)
    max_arcs_per_source: int = 3

    # Arcs generated in THIS timestep only (not cumulative)
    arcs: list[Arc] = field(default_factory=list)

    # Number of IMAGINE calls this timestep
    imagine_calls_this_dt: int = 0

    # Current imagine context
    imagine_context: str = ""

    # Linked list pointers
    prev: "ArcState | None" = None  # Link to previous timestep's state
    next: "ArcState | None" = None  # Link to next timestep's state

    @property
    def arcs_this_timestep(self) -> dict[str, int]:
        # Track num arcs committed per source in current timestep
        ret: dict[str, int] = {}
        for arc in self.arcs:
            source = arc.source
            if source not in ret:
                ret[source] = 0
            ret[source] += 1
        return ret

    def is_simulation_complete(self) -> bool:
        """Check if the simulation has reached or exceeded max_day/max_hour."""
        if self.current_day > self.max_day:
            return True
        if self.current_day == self.max_day and self.current_hour >= self.max_hour:
            return True
        return False

    def advance_timestep(self) -> "ArcState | None":
        """Create and return a new state for the next timestep.

        Returns:
            New ArcState linked to this one, or None if simulation is complete.
        """
        new_hour = self.current_hour + self.dt_hours
        new_day = self.current_day
        if new_hour >= 24:
            new_hour -= 24
            new_day += 1

        new_state = ArcState(
            current_day=new_day,
            current_hour=new_hour,
            dt_hours=self.dt_hours,
            max_day=self.max_day,
            max_hour=self.max_hour,
            max_arcs_per_source=self.max_arcs_per_source,
            arcs=[],
            imagine_calls_this_dt=0,
            imagine_context=self.imagine_context,  # Carry forward context
            prev=self,  # Link to previous state
        )
        self.next = new_state  # Link to next state
        
        # Check if new state is past the end
        if new_state.is_simulation_complete():
            return None
            
        return new_state

    def all_sources_saturated(self, source_ids: list[int]) -> bool:
        """Check if all sources have reached max arcs for this timestep."""
        return all(
            self.arcs_this_timestep.get(sid, 0) >= self.max_arcs_per_source
            for sid in source_ids
        )
    
    def get_all_arcs_up_to_now(self) -> list[Arc]:
        """Collect all arcs from the beginning of simulation up to current timestep.
        
        Returns:
            List of all arcs from all previous timesteps and current timestep.
        """
        all_arcs: list[Arc] = []
        current = self
        while current is not None:
            all_arcs.extend(current.arcs)
            current = current.prev
        return all_arcs

    def generate_dependency_arcs(self, past_horizon: int = 1) -> None:
        """Generate dependency arcs based on future forward arcs.

        A dependency arc (=>) is created when an object is the target of a forward arc
        in the current timestep AND is the source of a forward arc in a future timestep.
        This indicates that the future arc is "blocked" until the current arc completes.

        For example:
        - Timestep 0: obj1 -> obj2 (forward)
        - Timestep 1: obj2 -> obj3 (forward)
        - Result: At timestep 0, add obj2 => obj3 (dependency), with reason = arc_shortname of obj1->obj2

        Args:
            past_horizon: How many timesteps back to create dependency edges.
                         A value of 1 means timestep T+1's arcs create dependencies in timestep T.
        """
        # First, collect all states into a list (chronological order)
        states: list[ArcState] = []
        current = self
        while current is not None:
            states.append(current)
            current = current.prev
        states.reverse()  # Now states[0] is the earliest timestep

        # For each timestep, look at future timesteps within past_horizon
        for t, state in enumerate(states):
            # Look at future timesteps within the horizon
            for future_t in range(t + 1, min(t + 1 + past_horizon, len(states))):
                future_state = states[future_t]

                # For each forward arc in the current timestep
                for current_arc in state.arcs:
                    if current_arc.arc_type == "dependency":
                        continue  # Skip already-created dependency arcs

                    current_target = current_arc.target

                    # Check if this target is a source in any future arc
                    for future_arc in future_state.arcs:
                        if future_arc.arc_type == "dependency":
                            continue  # Skip dependency arcs

                        future_source = future_arc.source

                        if current_target == future_source:
                            # Create a dependency arc in the current timestep
                            # The dependency is: future_source => future_target is blocked
                            # The reason is the arc_shortname of the blocking arc (current_arc)
                            dependency_arc = Arc(
                                source=future_arc.source,
                                target=future_arc.target,
                                weight=future_arc.weight,
                                label=current_arc.arc_shortname,
                                arc_shortname=f"{future_arc.source}=>{future_arc.target}",
                                arc_type="dependency",
                                timestamp_hours=(state.current_day - 1) * 24 + state.current_hour,
                                day=state.current_day,
                                hour=state.current_hour,
                                blocked_by=current_arc.arc_shortname,
                            )
                            state.arcs.append(dependency_arc)

    def to_dict(self) -> dict[str, Any]:
        """Convert ArcState to a dictionary for JSON serialization.
        
        Note: The 'prev' pointer is not included; use to_dict_list() to serialize
        the full linked list.
        """
        return {
            "current_day": self.current_day,
            "current_hour": self.current_hour,
            "dt_hours": self.dt_hours,
            "max_day": self.max_day,
            "max_hour": self.max_hour,
            "max_arcs_per_source": self.max_arcs_per_source,
            "arcs": [arc.to_dict() for arc in self.arcs],
            "imagine_calls_this_dt": self.imagine_calls_this_dt,
            "imagine_context": self.imagine_context,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any], prev: "ArcState | None" = None, next: "ArcState | None" = None) -> "ArcState":
        """Create an ArcState from a dictionary.
        
        Args:
            data: Dictionary with ArcState fields.
            prev: Optional previous ArcState to link to.
            next: Optional next ArcState to link to.
            
        Returns:
            New ArcState instance.
        """
        return cls(
            current_day=data["current_day"],
            current_hour=data["current_hour"],
            dt_hours=data["dt_hours"],
            max_day=data["max_day"],
            max_hour=data["max_hour"],
            max_arcs_per_source=data["max_arcs_per_source"],
            arcs=[Arc.from_dict(arc) for arc in data["arcs"]],
            imagine_calls_this_dt=data.get("imagine_calls_this_dt", 0),
            imagine_context=data.get("imagine_context", ""),
            prev=prev,
            next=next,
        )

    def save(self, map_path: Path, output_filename: str = "arcs.json") -> Path:
        """Save the full ArcState linked list to arcs.json in the map directory.

        The output format is a list of ArcState dictionaries, ordered chronologically
        (earliest timestep first). Each ArcState includes all its fields and a list
        of Arc objects serialized as dictionaries.

        Args:
            map_path: Directory to save the arcs.json file.
            output_filename: Name of the output file.

        Returns:
            Path to the saved arcs.json file.
        """
        # Ensure the map directory exists
        map_path.mkdir(parents=True, exist_ok=True)
        arcs_path = map_path / output_filename

        def collect_states(state: ArcState | None) -> list[dict[str, Any]]:
            """Traverse the linked list and collect all states as dicts.

            Args:
                state: The head (most recent) state of the linked list.

            Returns:
                List of ArcState dicts, ordered chronologically (earliest first).
            """
            states: list[dict[str, Any]] = []
            current = state
            while current is not None:
                states.append(current.to_dict())
                current = current.prev
            # Reverse to get chronological order (earliest first)
            states.reverse()
            return states

        # Collect all states as dicts
        all_states = collect_states(self)

        with open(arcs_path, "w") as f:
            json.dump(all_states, f, indent=2)

        total_arcs = sum(len(s["arcs"]) for s in all_states)
        print(f"Saved {total_arcs} arcs across {len(all_states)} timesteps to {arcs_path}")
        return arcs_path

    @classmethod
    def load(cls, arcs_path: Path) -> "ArcState":
        """Load ArcState linked list from an arcs.json file.

        Args:
            arcs_path: Path to the arcs.json file.

        Returns:
            The final (most recent) ArcState, with prev pointers linking
            to earlier timesteps.
        """
        with open(arcs_path, "r") as f:
            all_states_data = json.load(f)
        
        if not all_states_data:
            raise ValueError(f"No states found in {arcs_path}")
        
        # Build the linked list from chronological order
        prev_state: ArcState | None = None
        states: list[ArcState] = []
        for state_data in all_states_data:
            current_state = cls.from_dict(state_data, prev=prev_state)
            if prev_state:
                prev_state.next = current_state
            states.append(current_state)
            prev_state = current_state
        
        # Return the final (most recent) state
        return prev_state  # type: ignore