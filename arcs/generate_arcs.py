"""Generate arcs (directed graph edges) between objects in a map.

This script procedurally generates arcs and writes them to arcs.json in the map directory.
Each arc connects a source object to a target object with an associated weight and label.

Usage:
    python generate_arcs.py                                  # Use defaults
    python generate_arcs.py generation=langchain             # Use langchain agent generation
    python generate_arcs.py sources=limited                  # Limit sources to 10
    python generate_arcs.py targets=random                   # Random number of targets
    python generate_arcs.py sources=blacklist targets=whitelist  # Different filters
    python generate_arcs.py sources.number_to_use.constant=5 # Override specific param
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any, Literal

import hydra
from hydra.utils import instantiate
from langchain.tools import tool
from langchain.agents import create_agent
from omegaconf import DictConfig, OmegaConf

from arcs.arc_state import Arc, ArcState


class AgentFinish(Exception):
    """Custom exception to signal agent completion with return values."""
    def __init__(self, return_values: dict[str, Any], log: str = ""):
        self.return_values = return_values
        self.log = log
        super().__init__(log)


class ArcGenerator:
    """Generate arcs (directed graph edges) between objects in a map."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the arc generator with configuration.
        
        Args:
            cfg: Hydra configuration with map_path, sources, targets, generation, etc.
        """
        def load_segment_annotations():
            """Load segment annotations from map file."""
            segments_anno_path = self.map_path / "segments_anno.json"
            if segments_anno_path.exists():
                with open(segments_anno_path, "r") as f:
                    return json.load(f)
            return {"segGroups": []}
        
        def build_objects_from_segments(segments_anno):
            """Build objects list from segment groups."""
            objects = []
            for idx, seg_group in enumerate(segments_anno.get("segGroups", [])):
                obj_id = seg_group.get("id", seg_group.get("object_id", idx))
                label = seg_group.get("label", seg_group.get("caption", f"object_{idx}"))
                objects.append({
                    "id": obj_id,
                    "label": label,
                    "object_id": obj_id,  # Include both naming conventions
                    "caption": label,
                })
            return objects
        
        def filter_objects(objects, config):
            """Filter and limit objects based on config (names and number constraints).
            
            Args:
                objects: List of object dictionaries to filter.
                config: Config section (cfg.sources or cfg.targets).
                
            Returns:
                Filtered list of objects.
            """
            filtered = objects.copy()
            
            # Filter by name inclusion/exclusion
            names_to_include = config.get("names_to_include")
            names_to_exclude = config.get("names_to_exclude")
            
            if names_to_include is not None:
                include_set = set(OmegaConf.to_container(names_to_include))
                filtered = [
                    obj for obj in filtered 
                    if obj.get("label", obj.get("caption", "")).lower() in {n.lower() for n in include_set}
                ]
            
            if names_to_exclude is not None:
                exclude_set = set(OmegaConf.to_container(names_to_exclude))
                filtered = [
                    obj for obj in filtered 
                    if obj.get("label", obj.get("caption", "")).lower() not in {n.lower() for n in exclude_set}
                ]
            
            # Apply number limit
            constant = config.number_to_use.get("constant")
            random_min = config.number_to_use.get("random_min")
            random_max = config.number_to_use.get("random_max")
            
            if constant is not None:
                num_to_use = constant
            elif random_min is not None and random_max is not None:
                num_to_use = random.randint(random_min, random_max)
            else:
                num_to_use = None
            
            if num_to_use is not None:
                filtered = filtered[:num_to_use]
            
            return filtered
        
        # Initialize paths and config
        self.cfg = cfg
        self.map_path = Path(cfg.map_path)
        self.arcs_output_dir = Path(cfg.arcs_output_dir)
        self.output_filename = cfg.output_filename
        
        # Setup LLM cache directory
        self.llm_cache_dir = self.arcs_output_dir / ".llm_cache"
        if cfg.get("delete_llm_cache", False) and self.llm_cache_dir.exists():
            print(f"Deleting LLM cache at {self.llm_cache_dir}")
            shutil.rmtree(self.llm_cache_dir)
        self.llm_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load segment annotations and build objects
        segments_anno = load_segment_annotations()
        self.all_objects = build_objects_from_segments(segments_anno)
        assert self.all_objects
        
        # Filter sources and targets
        self.source_objects = filter_objects(self.all_objects, cfg.sources)
        self.target_objects = filter_objects(self.all_objects, cfg.targets)
        
        # Dispatch table for generation methods
        self.generation_methods = {
            "dummy": self._generate_arcs_dummy,
            "langchain": self._generate_arcs_langchain,
        }
        
        # Simulation state (initialized when langchain generation is used)
        self.state: ArcState | None = None
        
        # Track if structured parsing has succeeded at least once
        self._structured_parsing_succeeded = False



    def _generate_arcs_dummy(
        self,
        source_objects: list[dict[str, Any]],
        target_objects: list[dict[str, Any]],
        all_objects: list[dict[str, Any]],
    ) -> list[Arc]:
        """Generate dummy arcs for testing.
        
        Args:
            source_objects: Filtered source objects.
            target_objects: Filtered target objects.
            all_objects: All object data.
            
        Returns:
            List of Arc instances.
        """
        arcs: list[Arc] = []
        if source_objects and target_objects:
            source_label = source_objects[0].get("label", source_objects[0].get("caption", "object_1"))
            target_label = target_objects[-1].get("label", target_objects[-1].get("caption", "object_15"))
            arcs.append(Arc(
                source=source_label,
                target=target_label,
                weight=0.8,
                label="related_to",
                arc_shortname=f"{source_label}->related_to->{target_label}",
                arc_type="forward",
                timestamp_hours=0.0,
                day=1,
                hour=0.0,
            ))
        else:
            # Fallback if no objects loaded
            arcs.append(Arc(
                source="object_1",
                target="object_15",
                weight=0.8,
                label="related_to",
                arc_shortname="object_1->related_to->object_15",
                arc_type="forward",
                timestamp_hours=0.0,
                day=1,
                hour=0.0,
            ))
        return arcs

    def _build_objects_str(self, objects: list[dict[str, Any]]) -> str:
        """Build a string description of objects for prompts."""
        descriptions = []
        for obj in objects:
            obj_id = obj.get("id", obj.get("object_id", "unknown"))
            label = obj.get("label", obj.get("caption", f"object_{obj_id}"))
            descriptions.append(f"  - ID {obj_id}: {label}")
        return "\n".join(descriptions) if descriptions else "  No objects available"

    def _get_cache_path(self, prompt: str, prefix: str = "llm") -> Path:
        """Get the cache file path for a given prompt.
        
        Args:
            prompt: The prompt to hash for the cache key.
            prefix: Prefix for the cache filename.
            
        Returns:
            Path to the cache file.
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return self.llm_cache_dir / f"{prefix}_{prompt_hash}.json"
    
    def _call_imagine(
        self,
        mode: Literal["long_horizon", "short_horizon"],
        num_called_this_dt: int,
    ) -> str:
        """Internal IMAGINE implementation using a separate LLM call.
        
        This provides grounded, realistic context about the scene - NOT fantasy or fringe events.
        Results are cached based on the prompt hash.
        
        Args:
            mode: Either "long_horizon" (multi-day planning) or "short_horizon" (current moment).
            num_called_this_dt: How many times IMAGINE has been called this timestep.
            
        Returns:
            A grounded description of scene events/patterns.
        """
        gen_cfg = self.cfg.generation
        
        objects_str = self._build_objects_str(self.all_objects)
        
        if mode == "long_horizon":
            prompt = gen_cfg.imagine_long_horizon_prompt.format(
                objects=objects_str,
                scene_description=gen_cfg.scene_description,
                days=gen_cfg.days,
            )
        else:
            prompt = gen_cfg.imagine_short_horizon_prompt.format(
                objects=objects_str,
                scene_description=gen_cfg.scene_description,
                dt=self.state.dt_hours if self.state else 0,
            )
        
        # Check cache
        cache_path = self._get_cache_path(prompt, prefix=f"imagine_{mode}")
        if cache_path.exists():
            print(f"Using cached IMAGINE result from {cache_path.name}")
            with open(cache_path, "r") as f:
                cached = json.load(f)
                return cached["content"]
        
        # Call LLM
        llm = instantiate(self.cfg.llm)
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Save to cache
        with open(cache_path, "w") as f:
            json.dump({"prompt": prompt, "content": content}, f, indent=2)
        print(f"Cached IMAGINE result to {cache_path.name}")
        
        return content

    def _parse_initial_hour_from_imagine(self, long_horizon_context: str) -> float:
        """Parse the long horizon IMAGINE result to determine the initial hour.
        
        First tries to parse the structured format directly, then falls back to LLM.
        Results are cached based on the prompt hash.
        
        Args:
            long_horizon_context: The result from the long horizon IMAGINE call.
            
        Returns:
            Initial hour (0-23) for the simulation.
        """
        import re
        
        # First, try to parse the structured format directly
        # Look for first time marker like "**6:00 AM" or "**12:00 PM"
        time_pattern = r'\*\*(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)'
        match = re.search(time_pattern, long_horizon_context)
        if match:
            hour = int(match.group(1))
            am_pm = match.group(3).upper()
            if am_pm == "PM" and hour != 12:
                hour += 12
            elif am_pm == "AM" and hour == 12:
                hour = 0
            print(f"Parsed initial hour from structured format: {hour}")
            return float(hour)
        
        # Also try simpler formats like "6:00 AM -" or "6:00AM:"
        simple_time_pattern = r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)'
        match = re.search(simple_time_pattern, long_horizon_context)
        if match:
            hour = int(match.group(1))
            am_pm = match.group(3).upper()
            if am_pm == "PM" and hour != 12:
                hour += 12
            elif am_pm == "AM" and hour == 12:
                hour = 0
            print(f"Parsed initial hour from simple time format: {hour}")
            return float(hour)
        
        # Fall back to LLM parsing
        prompt = f"""Based on the following scene timeline, determine what hour of the day (0-23) the simulation should start at.
        
Scene timeline:
{long_horizon_context}

Respond with only a number between 0 and 23 representing the starting hour. For example:
- If it starts in the morning, respond with a number like 6, 7, or 8
- If it starts at noon, respond with 12
- If it starts in the evening, respond with 18 or 19
- If it starts at midnight, respond with 0

Respond with ONLY the number, nothing else."""
        
        # Check cache
        cache_path = self._get_cache_path(prompt, prefix="initial_hour")
        if cache_path.exists():
            print(f"Using cached initial hour from {cache_path.name}")
            with open(cache_path, "r") as f:
                cached = json.load(f)
                return cached["hour"]
        
        # Call LLM
        llm = instantiate(self.cfg.llm)
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse the hour from the response
        try:
            # Extract first number from response
            match = re.search(r'\d+', content.strip())
            if match:
                hour = float(match.group())
                # Clamp to valid hour range
                hour = max(0.0, min(23.0, hour))
            else:
                print(f"Warning: Could not parse hour from LLM response: {content}. Using default 8.0")
                hour = 8.0  # Default to 8 AM
        except Exception as e:
            print(f"Warning: Error parsing initial hour: {e}. Using default 8.0")
            hour = 8.0  # Default to 8 AM
        
        # Save to cache
        with open(cache_path, "w") as f:
            json.dump({"prompt": prompt, "content": content, "hour": hour}, f, indent=2)
        print(f"Cached initial hour to {cache_path.name}")
        
        return hour

    def _parse_long_horizon_for_current_time(
        self,
        long_horizon_context: str,
        current_day: int,
        current_hour: float,
    ) -> tuple[str, bool]:
        """Extract the relevant section from long_horizon_context for the current day/hour.
        
        First tries to parse structured format with day/hour markers, then falls back
        to LLM extraction if the format is free-form.
        
        Args:
            long_horizon_context: The full long horizon IMAGINE result.
            current_day: Current simulation day (1-indexed).
            current_hour: Current hour (0-23).
            
        Returns:
            Tuple of (context_string, skip_this_timestep) where skip_this_timestep is True
            if there are no meaningful arcs/interactions at this timestep.
        """
        import re
        
        # Try structured parsing first
        parsed_context = self._try_parse_structured_context(
            long_horizon_context, current_day, current_hour
        )
        if parsed_context is not None:
            # Structured parsing found content for this timestep
            return (parsed_context, False)
        elif self._structured_parsing_succeeded:
            # Structured parsing has worked before but found nothing for this timestep
            # This is a real gap in the schedule - skip this timestep
            return (None, True)
        else:
            # Structured parsing has never worked - fall back to LLM extraction
            context = self._extract_context_with_llm(
                long_horizon_context, current_day, current_hour
            )
            # LLM returns None if we should skip
            skip_this_timestep = context is None
            return (context, skip_this_timestep)
    
    def _try_parse_structured_context(
        self,
        long_horizon_context: str,
        current_day: int,
        current_hour: float,
    ) -> str | None:
        """Try to parse structured day/hour format from long_horizon_context.
        
        Expected format:
        ### Day 1
        **6:00 AM - 7:00 AM: Activity Title**
        - Description...
        
        Args:
            long_horizon_context: The full long horizon context.
            current_day: Current day (1-indexed).
            current_hour: Current hour (0-23).
            
        Returns:
            Extracted context string, or None if parsing fails.
        """
        import re
        
        def parse_time_to_hour(time_str: str) -> float | None:
            """Parse time string like '6:00 AM' to hour (0-23)."""
            match = re.match(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)', time_str.strip())
            if not match:
                return None
            hour = int(match.group(1))
            am_pm = match.group(3).upper()
            if am_pm == "PM" and hour != 12:
                hour += 12
            elif am_pm == "AM" and hour == 12:
                hour = 0
            return float(hour)
        
        # Split by day markers
        day_pattern = r'###\s*Day\s*(\d+)'
        day_splits = re.split(day_pattern, long_horizon_context)
        
        # day_splits will be: [preamble, day_num, content, day_num, content, ...]
        if len(day_splits) < 3:
            return None  # No structured days found
        
        # Build a dict of day -> content
        days_content = {}
        for i in range(1, len(day_splits) - 1, 2):
            try:
                day_num = int(day_splits[i])
                content = day_splits[i + 1]
                days_content[day_num] = content
            except (ValueError, IndexError):
                continue
        
        if current_day not in days_content:
            return None
        
        # Mark that we found structured day content
        self._structured_parsing_succeeded = True
        day_content = days_content[current_day]
        
        # Parse time blocks within this day
        # A time block header looks like: **6:00 AM - 7:00 AM: Title**
        # The body continues until the next time block header or end
        # Use a pattern that matches time at start, then anything until closing **
        time_header_pattern = r'\*\*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)(?:\s*-\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))?[^*]*?)\*\*'
        
        # Find all header positions
        headers = list(re.finditer(time_header_pattern, day_content))
        
        if not headers:
            # No time blocks found, return the whole day content
            return f"Day {current_day}:\n{day_content.strip()}"
        
        # Build blocks: each block is (header_text, body_text)
        blocks = []
        for i, match in enumerate(headers):
            header_text = match.group(1)
            body_start = match.end()
            # Body ends at the next header or end of content
            if i + 1 < len(headers):
                body_end = headers[i + 1].start()
            else:
                body_end = len(day_content)
            body_text = day_content[body_start:body_end].strip()
            blocks.append((header_text, body_text))
        
        if not blocks:
            # No time blocks found, return the whole day content
            return f"Day {current_day}:\n{day_content.strip()}"
        
        # Find the block(s) that include the current hour
        relevant_blocks = []
        for header, body in blocks:
            # Try to parse time range from header
            time_range_match = re.search(
                r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\s*-\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))',
                header
            )
            if time_range_match:
                start_hour = parse_time_to_hour(time_range_match.group(1))
                end_hour = parse_time_to_hour(time_range_match.group(2))
                if start_hour is not None and end_hour is not None:
                    # Check if current_hour falls within this range
                    if start_hour <= current_hour < end_hour:
                        relevant_blocks.append(f"**{header.strip()}**\n{body.strip()}")
            else:
                # Try single time match
                single_time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', header)
                if single_time_match:
                    block_hour = parse_time_to_hour(single_time_match.group(1))
                    # Only include if it's an exact or very close match
                    if block_hour is not None and block_hour == current_hour:
                        relevant_blocks.append(f"**{header.strip()}**\n{body.strip()}")
        
        if relevant_blocks:
            self._structured_parsing_succeeded = True
            return f"Day {current_day}, Hour {current_hour:.0f}:00:\n\n" + "\n\n".join(relevant_blocks)
        
        # No specific time match found - this means no timestep was explicitly specified
        # for this hour, so we should return None to signal it should be skipped
        return None
    
    def _extract_context_with_llm(
        self,
        long_horizon_context: str,
        current_day: int,
        current_hour: float,
    ) -> str | None:
        """Use LLM to extract relevant context when structured parsing fails.
        
        Args:
            long_horizon_context: The full long horizon context.
            current_day: Current day (1-indexed).
            current_hour: Current hour (0-23).
            
        Returns:
            LLM-extracted relevant context, or None if timestep should be skipped.
        """
        import re
        
        # Format hour nicely
        hour_12 = current_hour % 12
        if hour_12 == 0:
            hour_12 = 12
        am_pm = "AM" if current_hour < 12 else "PM"
        time_str = f"{int(hour_12)}:00 {am_pm}"
        
        prompt = f"""Extract the relevant section from this scene timeline for Day {current_day} at approximately {time_str} ({int(current_hour)}:00 in 24-hour format).

Full timeline:
{long_horizon_context}

Provide ONLY the relevant activities and context for this specific time period. Include:
1. What is currently happening at this time
2. Any activities that just finished (for context)
3. Any activities about to start

If there are NO meaningful activities or object interactions at this time (e.g., everyone is sleeping, the space is empty/closed), respond with exactly "SKIP" and nothing else.

Otherwise, write a clean, concise summary focused on Day {current_day}, {time_str}."""
        
        # Check cache
        cache_path = self._get_cache_path(prompt, prefix="context_extract")
        if cache_path.exists():
            print(f"Using cached context extraction from {cache_path.name}")
            with open(cache_path, "r") as f:
                cached = json.load(f)
                content = cached["content"]
                if content.strip().upper() == "SKIP":
                    return None
                return content
        
        # Call LLM
        llm = instantiate(self.cfg.llm)
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Save to cache
        with open(cache_path, "w") as f:
            json.dump({"prompt": prompt, "content": content}, f, indent=2)
        print(f"Cached context extraction to {cache_path.name}")
        
        # Check if LLM indicated we should skip
        if content.strip().upper() == "SKIP":
            return None
        
        return content
    
    def _create_tools(self) -> list:
        """Create the langchain tools for arc generation."""
        
        # References to self for closures
        generator = self
        
        @tool
        def imagine(mode: str = "short_horizon") -> str:
            """Get grounded context about what might happen in the scene.
            
            This tool provides realistic, everyday scenarios based on the objects present.
            It does NOT generate unlikely accidents, fringe events, or fantasy scenarios.
            
            Args:
                mode: The imagination mode (automatically determined by system).
                
            Returns:
                A description of realistic events and patterns in the scene.
            """
            if generator.state is None:
                return "Error: Simulation state not initialized"
            
            generator.state.imagine_calls_this_dt += 1
            
            # Mode is controlled by the system, not the agent
            # Agent always sees it as the same tool
            actual_mode: Literal["long_horizon", "short_horizon"] = "short_horizon"
            
            context = generator._call_imagine(
                mode=actual_mode,
                num_called_this_dt=generator.state.imagine_calls_this_dt,
            )
            generator.state.imagine_context = context
            return context
        
        @tool
        def commit_arc(source: int, target: int, reason: str, priority: float, arc_shortname: str) -> str:
            """Commit a single arc (directed relationship) between two objects.
            
            Each arc represents a meaningful interaction or relationship between objects
            at the current timestep.
            
            Args:
                source: The ID of the source object.
                target: The ID of the target object.
                reason: A brief description of why these objects are related.
                priority: Importance of this relationship (0.0 to 1.0).
                arc_shortname: A short identifier for this arc (e.g., "person_uses_computer").
                
            Returns:
                Confirmation message or error.
            """
            def find_object_label(obj_id: int, objects: list[dict], obj_type: str) -> tuple[str | None, str | None]:
                """Find object by ID and return its label, or None with error message.
                
                Returns:
                    Tuple of (label, error_message). One will be None.
                """
                for obj in objects:
                    current_id = obj.get("id", obj.get("object_id"))
                    if current_id == obj_id:
                        label = obj.get("label", obj.get("caption", f"object_{obj_id}"))
                        return (label, None)
                
                # Not found - generate error message
                available_ids = [obj.get("id", obj.get("object_id")) for obj in objects]
                error = f"Error: {obj_type.capitalize()} {obj_id} not in available {obj_type}s: {available_ids}"
                return (None, error)
            
            if generator.state is None:
                return "Error: Simulation state not initialized"
            
            state = generator.state
            
            # Check if source has reached max arcs
            current_count = state.arcs_this_timestep.get(source, 0)
            if current_count >= state.max_arcs_per_source:
                return f"Error: Source {source} has reached max arcs ({state.max_arcs_per_source}) for this timestep"
            
            # Validate source and target exist, and fetch their labels
            source_label, source_error = find_object_label(source, generator.source_objects, "source")
            if source_error:
                return source_error
            
            target_label, target_error = find_object_label(target, generator.target_objects, "target")
            if target_error:
                return target_error
            
            # Check for duplicate and overlapping arcs across all timesteps
            all_arcs = state.get_all_arcs_up_to_now()
            for existing_arc in all_arcs:
                # Skip dependency arcs in overlap detection
                if existing_arc.arc_type == "dependency":
                    continue
                    
                # Check for exact duplicate (same source, target, and reason)
                if (existing_arc.source == source_label and 
                    existing_arc.target == target_label and 
                    existing_arc.label == reason):
                    return f"Error: Duplicate arc rejected. Arc with same source ({source_label}), target ({target_label}), and reason ({reason}) already exists."
                
                # Check for bidirectional/reverse arc (S->T when T->S exists, or vice versa)
                if (existing_arc.source == target_label and 
                    existing_arc.target == source_label):
                    return f"Error: Overlapping bidirectional arc rejected. Reverse arc {target_label} -> {source_label} already exists (reason: {existing_arc.label})."
            
            # Create and store the arc using Arc class
            arc = Arc(
                source=source_label,
                target=target_label,
                weight=max(0.0, min(1.0, priority)),  # Clamp to [0, 1]
                label=reason,
                arc_shortname=arc_shortname,
                arc_type="forward",
                timestamp_hours=(state.current_day - 1) * 24 + state.current_hour,
                day=state.current_day,
                hour=state.current_hour,
            )
            state.arcs.append(arc)
            
            # Check if all sources are saturated - advance timestep immediately and exit agent
            source_ids = [obj.get("id", obj.get("object_id")) for obj in generator.source_objects]
            if state.all_sources_saturated(source_ids):
                arcs_this_step = sum(state.arcs_this_timestep.values())
                raise AgentFinish(
                    return_values={"output": f"Arc committed: {source_label} -> {target_label} ({reason}). All sources saturated. Generated {arcs_this_step} arcs this timestep. Advancing to next timestep."},
                    log=f"All sources saturated after committing arc. Forcing timestep advance."
                )
            
            return f"Arc committed: {source_label} -> {target_label} ({reason}). Priority: {priority:.2f}"
        
        @tool
        def next_timestamp() -> str:
            """Advance to the next timestep in the simulation.
            
            Call this when you have finished generating arcs for the current timestep.
            
            Returns:
                Status message about the new timestep or simulation end.
            """
            if generator.state is None:
                return "Error: Simulation state not initialized"
            
            state = generator.state
            
            arcs_this_step = sum(state.arcs_this_timestep.values())
            raise AgentFinish(
                return_values={"output": f"Timestep complete. Generated {arcs_this_step} arcs. Advancing to next timestep."},
                log=f"Agent requested timestep advance via next_timestamp."
            )
        
        return [imagine, commit_arc, next_timestamp]

    def _generate_arcs_langchain(
        self,
        source_objects: list[dict[str, Any]],
        target_objects: list[dict[str, Any]],
        all_objects: list[dict[str, Any]],
    ) -> ArcState:
        """Generate arcs using a langchain agent with tools.
        
        The agent uses IMAGINE, COMMIT_ARC, and NEXT_TIMESTAMP tools to iteratively
        build arcs over a simulated time period.
        
        Args:
            source_objects: Filtered source objects.
            target_objects: Filtered target objects.
            all_objects: All object data.
            
        Returns:
            The final ArcState (head of the linked list of all timesteps).
        """
        gen_cfg = self.cfg.generation
        
        # Initialize simulation state
        days = gen_cfg.days
        dt_hours = gen_cfg.dt_hours
        
        self.state = ArcState(
            dt_hours=dt_hours,
            max_day=days + 1,  # End at the start of day (days+1)
            max_hour=0.0,
            max_arcs_per_source=gen_cfg.max_arcs_per_source,
        )
        
        # Instantiate the LLM
        llm = instantiate(self.cfg.llm)
        
        # Create tools
        tools = self._create_tools()
        
        # Step 1: Force IMAGINE with long horizon mode to get initial context
        print("Generating long-horizon scene context...")
        long_horizon_context = self._call_imagine("long_horizon", 0)
        #self.state.imagine_context = long_horizon_context
        print(f"Long-horizon context:\n{long_horizon_context[:500]}...")
        
        # Step 2: Parse the initial hour from the long horizon context
        print("\nDetermining initial hour from scene context...")
        initial_hour = self._parse_initial_hour_from_imagine(long_horizon_context)
        self.state.current_hour = initial_hour
        print(f"Initial hour set to: {initial_hour}")
        
        # Get the system prompt from config
        system_prompt = gen_cfg.agent_system_prompt
        
        # Create the agent using the new create_agent API
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )
        
        # Build object strings for the prompt
        source_obj_str = self._build_objects_str(source_objects)
        target_obj_str = self._build_objects_str(target_objects)
        
        # Step 3: Run the simulation loop
        print(f"\nStarting simulation: {days} days, {dt_hours}h timesteps")
        
        while self.state is not None and not self.state.is_simulation_complete():
            # Retrieve the relevant part of the long_horizon_context for the current day/hour
            local_context, skip_this_timestep = self._parse_long_horizon_for_current_time(
                long_horizon_context,
                self.state.current_day,
                self.state.current_hour,
            )
            self.state.imagine_context = local_context

            print(f"\n--- Timestep: Day {self.state.current_day}, Hour {self.state.current_hour:.1f} ---")
            
            if skip_this_timestep:
                print("Skipping timestep: No meaningful interactions expected at this time.")
            else:
                # Format the iteration prompt with local context only
                iteration_input = gen_cfg.agent_iteration_prompt.format(
                    current_day=self.state.current_day,
                    current_hour=self.state.current_hour,
                    max_day=self.state.max_day,
                    max_hour=self.state.max_hour,
                    imagine_context=local_context,
                    source_objects=source_obj_str,
                    target_objects=target_obj_str,
                )
                
                try:
                    # Run agent for this timestep using the new invoke format
                    result = agent.invoke({
                        "messages": [{"role": "user", "content": iteration_input}]
                    })
                    # Extract output from the new response format
                    messages = result.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        output = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                        print(f"Agent output: {output[:200]}")
                    else:
                        print("Agent output: No messages returned")
                except AgentFinish as af:
                    # Agent signaled completion via AgentFinish (from commit_arc or next_timestamp)
                    output = af.return_values.get("output", "Timestep complete")
                    print(f"Agent finished: {output}")
                except Exception as e:
                    print(f"Agent error: {e}")
            
            # Advance timestep - creates new state linked to current
            next_state = self.state.advance_timestep()
            if next_state is None:
                break
            self.state = next_state
        
        # Generate dependency arcs as a postprocessing step
        print("\nGenerating dependency arcs...")
        dependency_horizon = gen_cfg.get("dependency_horizon", 1)
        self.state.generate_dependency_arcs(past_horizon=dependency_horizon)
        
        # Count total arcs by traversing the linked list
        total_arcs = self._count_total_arcs()
        print(f"\nSimulation complete. Generated {total_arcs} total arcs across all timesteps.")
        return self.state

    def _count_total_arcs(self) -> int:
        """Count total arcs by traversing the linked list of states."""
        total = 0
        current = self.state
        while current is not None:
            total += len(current.arcs)
            current = current.prev
        return total



    def generate(self) -> ArcState:
        """Generate arcs between objects based on configuration.
        
        Returns:
            The final SimulationState (head of the linked list of all timesteps).
        """
        print(f"Using {len(self.source_objects)} sources and {len(self.target_objects)} targets")
        
        # Get generation method
        method_name = self.cfg.generation.name
        if method_name not in self.generation_methods:
            raise ValueError(f"Unknown generation method: {method_name}. Available: {list(self.generation_methods.keys())}")
        
        generate_fn = self.generation_methods[method_name]
        result = generate_fn(self.source_objects, self.target_objects, self.all_objects)
        
        # For backward compatibility with dummy generator that returns list
        if isinstance(result, list):
            # Wrap in a single-timestep state
            self.state = ArcState(arcs=result)
            return self.state
        
        return result


@hydra.main(version_base=None, config_path="conf_arcs", config_name="arcs")
def main(cfg: DictConfig) -> None:
    """Generate and save arcs to the map directory.
    
    Args:
        cfg: Hydra configuration.
    """
    print(OmegaConf.to_yaml(cfg))
    
    generator = instantiate({"_target_": "generate_arcs.ArcGenerator", "cfg": cfg}, _recursive_=False)
    state = generator.generate()
    state.save(generator.map_path, generator.output_filename)


if __name__ == "__main__":
    main()
