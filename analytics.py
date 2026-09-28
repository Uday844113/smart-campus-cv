class AttendanceAnalytics:

    def __init__(
        self,
        line_position=0.65,
        entry_direction="top_to_bottom"
    ):
        self.line_position = line_position
        self.entry_direction = entry_direction

        self.previous_positions = {}

        self.unique_entries = set()

        self.last_crossing_frame = {}

        self.frame_number = 0

        self.min_movement = 8

        self.crossing_cooldown = 20

        self.last_event = "Waiting..."

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        self.previous_positions.clear()

        self.unique_entries.clear()

        self.last_crossing_frame.clear()

        self.frame_number = 0

        self.last_event = "Waiting..."

    # =====================================================
    # UPDATE
    # =====================================================

    def update(self, frame, boxes):

        self.frame_number += 1

        frame_height = frame.shape[0]

        line_y = int(
            frame_height * self.line_position
        )

        current_ids = set()

        self.last_event = "No new entry/exit"

        # =================================================
        # NO DETECTIONS
        # =================================================

        if boxes is None or boxes.id is None:

            return (
                0,
                len(self.unique_entries),
                self.last_event,
                current_ids,
                line_y
            )

        # =================================================
        # GET TRACK IDS
        # =================================================

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )

        xyxy = (
            boxes.xyxy
            .cpu()
            .tolist()
        )

        # =================================================
        # PROCESS PEOPLE
        # =================================================

        for track_id, box in zip(
            track_ids,
            xyxy
        ):

            x1, y1, x2, y2 = box

            center_y = int(
                (y1 + y2) / 2
            )

            current_ids.add(track_id)

            # ---------------------------------------------
            # PREVIOUS POSITION
            # ---------------------------------------------

            previous_y = self.previous_positions.get(
                track_id
            )

            self.previous_positions[
                track_id
            ] = center_y

            if previous_y is None:
                continue

            # ---------------------------------------------
            # MOVEMENT
            # ---------------------------------------------

            movement = abs(
                center_y - previous_y
            )

            if movement < self.min_movement:
                continue

            # ---------------------------------------------
            # CROSSING COOLDOWN
            # ---------------------------------------------

            last_crossing = (
                self.last_crossing_frame.get(
                    track_id,
                    -999999
                )
            )

            if (
                self.frame_number
                - last_crossing
                < self.crossing_cooldown
            ):
                continue

            # ---------------------------------------------
            # CROSSING CHECK
            # ---------------------------------------------

            crossed_down = (
                previous_y < line_y
                and center_y >= line_y
            )

            crossed_up = (
                previous_y > line_y
                and center_y <= line_y
            )

            # =================================================
            # TOP -> BOTTOM
            # =================================================

            if self.entry_direction == "top_to_bottom":

                # Entry
                if crossed_down:

                    self.unique_entries.add(
                        track_id
                    )

                    self.last_crossing_frame[
                        track_id
                    ] = self.frame_number

                    self.last_event = (
                        f"Entry detected — ID {track_id}"
                    )

                # Exit
                elif crossed_up:

                    self.last_crossing_frame[
                        track_id
                    ] = self.frame_number

                    self.last_event = (
                        f"Exit detected — ID {track_id}"
                    )

            # =================================================
            # BOTTOM -> TOP
            # =================================================

            else:

                # Entry
                if crossed_up:

                    self.unique_entries.add(
                        track_id
                    )

                    self.last_crossing_frame[
                        track_id
                    ] = self.frame_number

                    self.last_event = (
                        f"Entry detected — ID {track_id}"
                    )

                # Exit
                elif crossed_down:

                    self.last_crossing_frame[
                        track_id
                    ] = self.frame_number

                    self.last_event = (
                        f"Exit detected — ID {track_id}"
                    )

        # =================================================
        # CURRENTLY PRESENT
        # =================================================

        currently_present = len(
            current_ids
        )

        return (
            currently_present,
            len(self.unique_entries),
            self.last_event,
            current_ids,
            line_y
        )