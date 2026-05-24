"""Measurement-note extraction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py


@dataclass(frozen=True)
class MeasurementNotes:
    hdf5_path: Path
    driver_name: str
    group_name: str
    angles_deg: tuple[int, ...]
    notes_by_angle: dict[int, str]
    titles_by_angle: dict[int, str]
    timing_corrected_by_angle: dict[int, bool]
    timing_offset_ms_by_angle: dict[int, float]
    timing_peak_time_ms_by_angle: dict[int, float]
    timing_global_peak_time_ms_by_angle: dict[int, float]
    timing_earliest_10pct_peak_time_ms_by_angle: dict[int, float]
    timing_first_strong_near_ref_lobe_time_ms_by_angle: dict[int, float]
    timing_selected_minus_first_strong_near_ref_lobe_ms_by_angle: dict[int, float]
    timing_selected_minus_first_strong_near_ref_lobe_path_mm_by_angle: dict[int, float]
    timing_selected_is_first_strong_near_ref_lobe_by_angle: dict[int, bool]
    timing_mdat_window_ref_minus_first_strong_near_ref_lobe_path_mm_by_angle: dict[int, float]
    timing_mdat_window_ref_is_first_strong_near_ref_lobe_by_angle: dict[int, bool]
    timing_mdat_window_ref_not_first_strong_near_ref_lobe_by_angle: dict[int, bool]
    timing_late_window_peak_time_ms_by_angle: dict[int, float]
    timing_late_window_peak_abs_rel_to_global_by_angle: dict[int, float]
    timing_late_window_peak_warning_by_angle: dict[int, bool]
    timing_peak_interpretation_by_angle: dict[int, str]
    timing_peak_selection_reason_by_angle: dict[int, str]
    timing_peak_policy_by_angle: dict[int, str]
    timing_current_loader_peak_rejected_by_angle: dict[int, bool]
    timing_current_loader_selected_early_event_by_angle: dict[int, bool]
    timing_suspicious_window_ref_alignment_by_angle: dict[int, bool]
    timing_suspicious_reflection_alignment_by_angle: dict[int, bool]
    target_kind: str
    diagnostic_only: bool
    not_acceptance_target: bool
    validation_hypothesis: str
    processing_policy: str
    peak_selection_policy: str
    gate_window_policy: str
    normalization_policy: str
    published_polar_explorer_path: str
    published_polar_explorer_url: str
    published_explorer_match: bool
    published_explorer_match_frequency_hz: float
    published_explorer_hdf5_frequency_hz: float
    published_explorer_label_frequency_hz: float
    published_explorer_match_max_abs_delta_db_1004_hz: float
    published_explorer_match_max_abs_delta_angle_deg_1004_hz: float
    parsed_distance_m: float | None
    parsed_height_m: float | None
    parsed_height_reference: str | None
    passive_state_status: str
    passive_state_evidence: str
    passive_state_acceptance_use: str
    passive_state_metadata_policy: str

    @property
    def unique_notes(self) -> tuple[str, ...]:
        notes = []
        seen = set()
        for angle in self.angles_deg:
            note = self.notes_by_angle.get(angle, "").strip()
            if note and note not in seen:
                notes.append(note)
                seen.add(note)
        return tuple(notes)

    @property
    def summary(self) -> str:
        distance = "unknown" if self.parsed_distance_m is None else f"{self.parsed_distance_m:.3g} m"
        if self.parsed_height_m is not None:
            height = f"{self.parsed_height_m:.3g} m"
        elif self.parsed_height_reference is not None:
            height = f"reference {self.parsed_height_reference}"
        else:
            height = "unknown"
        note = self.unique_notes[0].replace("\n", " / ") if self.unique_notes else "no notes"
        return f"distance={distance}; height={height}; first note={note}"

    @property
    def acceptance_target_allowed(self) -> bool:
        return not (self.diagnostic_only or self.not_acceptance_target)

    @property
    def acceptance_target_reason(self) -> str:
        if self.not_acceptance_target:
            return "target HDF5 is explicitly marked not_acceptance_target"
        if self.diagnostic_only:
            return "target HDF5 is explicitly marked diagnostic_only"
        if self.target_kind:
            return f"target kind `{self.target_kind}` is not marked diagnostic-only"
        return "target HDF5 is not marked diagnostic-only"

    @property
    def published_parity_target_allowed(self) -> bool:
        return (
            self.acceptance_target_allowed
            and self.target_kind == "andres_published_parity"
            and self.validation_hypothesis == "andres_published_parity"
            and self.published_explorer_match
        )

    @property
    def published_parity_target_reason(self) -> str:
        if not self.acceptance_target_allowed:
            return self.acceptance_target_reason
        if self.target_kind != "andres_published_parity":
            return f"target kind `{self.target_kind or 'unset'}` is not `andres_published_parity`"
        if self.validation_hypothesis != "andres_published_parity":
            return (
                f"validation hypothesis `{self.validation_hypothesis or 'unset'}` "
                "is not `andres_published_parity`"
            )
        if not self.published_explorer_match:
            return "target HDF5 has not been proven to match the published polar explorer"
        return "target HDF5 matches the published polar explorer processing/export"

    @property
    def suspicious_timing_angles_deg(self) -> tuple[int, ...]:
        """Angles whose HDF5 timing metadata indicates an unsafe IR-start alignment."""

        suspicious = []
        for angle in self.angles_deg:
            notes = self.notes_by_angle.get(angle, "")
            corrected = self.timing_corrected_by_angle.get(angle, False)
            offset_ms = self.timing_offset_ms_by_angle.get(angle, 0.0)
            audit_flag = self.timing_suspicious_reflection_alignment_by_angle.get(angle, False)
            if audit_flag or (corrected and "IR start time" in notes and abs(offset_ms) > 1.0):
                suspicious.append(angle)
        return tuple(suspicious)

    @property
    def direct_arrival_timing_unsafe_angles_deg(self) -> tuple[int, ...]:
        """Angles whose audited direct-arrival timing anchor should not be used for acceptance.

        This is broader than the legacy offset check: it also treats
        current-loader peak rejection, current-loader early-event selection,
        saved-window early-event alignment, and selected-vs-first-lobe ambiguity
        as unsafe. Late in-window peaks are reported separately via
        ``late_window_warning_angles_deg`` because they can contaminate the
        gated response without being the selected direct-arrival timing anchor.
        """

        unsafe = set(self.suspicious_timing_angles_deg)
        for angle in self.angles_deg:
            if self.timing_current_loader_peak_rejected_by_angle.get(angle, False):
                unsafe.add(angle)
            if self.timing_current_loader_selected_early_event_by_angle.get(angle, False):
                unsafe.add(angle)
            if self.timing_suspicious_window_ref_alignment_by_angle.get(angle, False):
                unsafe.add(angle)
            if self.timing_selected_not_first_lobe_by_angle(angle):
                unsafe.add(angle)
            if self.timing_peak_policy_unsafe_by_angle(angle):
                unsafe.add(angle)
        return tuple(sorted(unsafe))

    def timing_selected_not_first_lobe_by_angle(self, angle: int) -> bool:
        if angle not in self.timing_selected_is_first_strong_near_ref_lobe_by_angle:
            return False
        first_lobe_ms = self.timing_first_strong_near_ref_lobe_time_ms_by_angle.get(angle, float("nan"))
        return (
            first_lobe_ms == first_lobe_ms
            and not self.timing_selected_is_first_strong_near_ref_lobe_by_angle.get(angle, True)
        )

    @property
    def selected_not_first_lobe_angles_deg(self) -> tuple[int, ...]:
        return tuple(
            angle for angle in self.angles_deg if self.timing_selected_not_first_lobe_by_angle(angle)
        )

    @property
    def mdat_window_ref_not_first_lobe_angles_deg(self) -> tuple[int, ...]:
        return tuple(
            angle
            for angle in self.angles_deg
            if self.timing_mdat_window_ref_not_first_strong_near_ref_lobe_by_angle.get(angle, False)
        )

    def timing_peak_policy_unsafe_by_angle(self, angle: int) -> bool:
        policy = self.timing_peak_policy_by_angle.get(angle, "").strip().lower().replace("_", "-")
        if not policy or policy in {"first-strong", "ir-start"}:
            return False
        if policy == "strongest":
            return not self.timing_selected_is_first_strong_near_ref_lobe_by_angle.get(angle, False)
        return True

    @property
    def peak_policy_unsafe_angles_deg(self) -> tuple[int, ...]:
        return tuple(
            angle for angle in self.angles_deg if self.timing_peak_policy_unsafe_by_angle(angle)
        )

    @property
    def late_window_warning_angles_deg(self) -> tuple[int, ...]:
        return tuple(
            angle for angle in self.angles_deg if self.timing_late_window_peak_warning_by_angle.get(angle, False)
        )

    @property
    def peak_rejected_angles_deg(self) -> tuple[int, ...]:
        return tuple(
            angle
            for angle in self.angles_deg
            if self.timing_current_loader_peak_rejected_by_angle.get(angle, False)
        )

    @property
    def peak_selected_early_event_angles_deg(self) -> tuple[int, ...]:
        return tuple(
            angle
            for angle in self.angles_deg
            if self.timing_current_loader_selected_early_event_by_angle.get(angle, False)
        )

    @property
    def suspicious_window_ref_angles_deg(self) -> tuple[int, ...]:
        return tuple(
            angle
            for angle in self.angles_deg
            if self.timing_suspicious_window_ref_alignment_by_angle.get(angle, False)
        )

    @property
    def ir_start_note_angles_deg(self) -> tuple[int, ...]:
        """Angles whose REW notes say the trace used IR-start timing.

        This is a caveat marker, not proof that the processed HDF5 response is
        aligned to that early event. The raw mdat/window audit decides whether
        the processed response actually follows the early peak.
        """

        angles = []
        for angle in self.angles_deg:
            notes = self.notes_by_angle.get(angle, "")
            delay_ms = parse_delay_note_ms(notes)
            if "IR start time" in notes and (delay_ms is None or abs(delay_ms) > 1.0):
                angles.append(angle)
        return tuple(angles)


def load_measurement_notes(path: Path, driver_name: str, group_name: str = "angles") -> MeasurementNotes:
    notes_by_angle: dict[int, str] = {}
    titles_by_angle: dict[int, str] = {}
    timing_corrected_by_angle: dict[int, bool] = {}
    timing_offset_ms_by_angle: dict[int, float] = {}
    timing_peak_time_ms_by_angle: dict[int, float] = {}
    timing_global_peak_time_ms_by_angle: dict[int, float] = {}
    timing_earliest_10pct_peak_time_ms_by_angle: dict[int, float] = {}
    timing_first_strong_near_ref_lobe_time_ms_by_angle: dict[int, float] = {}
    timing_selected_minus_first_strong_near_ref_lobe_ms_by_angle: dict[int, float] = {}
    timing_selected_minus_first_strong_near_ref_lobe_path_mm_by_angle: dict[int, float] = {}
    timing_selected_is_first_strong_near_ref_lobe_by_angle: dict[int, bool] = {}
    timing_mdat_window_ref_minus_first_strong_near_ref_lobe_path_mm_by_angle: dict[int, float] = {}
    timing_mdat_window_ref_is_first_strong_near_ref_lobe_by_angle: dict[int, bool] = {}
    timing_mdat_window_ref_not_first_strong_near_ref_lobe_by_angle: dict[int, bool] = {}
    timing_late_window_peak_time_ms_by_angle: dict[int, float] = {}
    timing_late_window_peak_abs_rel_to_global_by_angle: dict[int, float] = {}
    timing_late_window_peak_warning_by_angle: dict[int, bool] = {}
    timing_peak_interpretation_by_angle: dict[int, str] = {}
    timing_peak_selection_reason_by_angle: dict[int, str] = {}
    timing_peak_policy_by_angle: dict[int, str] = {}
    timing_current_loader_peak_rejected_by_angle: dict[int, bool] = {}
    timing_current_loader_selected_early_event_by_angle: dict[int, bool] = {}
    timing_suspicious_window_ref_alignment_by_angle: dict[int, bool] = {}
    timing_suspicious_reflection_alignment_by_angle: dict[int, bool] = {}
    distance_attrs: list[float] = []
    height_attrs: list[float] = []
    height_refs: list[str] = []
    with h5py.File(path, "r") as h5:
        root_attrs = h5.attrs
        target_kind = _str_attr(root_attrs.get("target_kind")) or ""
        diagnostic_only = _bool_attr(root_attrs.get("diagnostic_only", False))
        not_acceptance_target = _bool_attr(root_attrs.get("not_acceptance_target", False))
        validation_hypothesis = _str_attr(root_attrs.get("validation_hypothesis")) or ""
        processing_policy = _text_attr(root_attrs.get("processing_policy")) or ""
        peak_selection_policy = _text_attr(root_attrs.get("peak_selection_policy")) or ""
        gate_window_policy = _text_attr(root_attrs.get("gate_window_policy")) or ""
        normalization_policy = _text_attr(root_attrs.get("normalization_policy")) or ""
        published_polar_explorer_path = _text_attr(root_attrs.get("published_polar_explorer_path")) or ""
        published_polar_explorer_url = _text_attr(root_attrs.get("published_polar_explorer_url")) or ""
        published_explorer_match = _bool_attr(root_attrs.get("published_explorer_match", False))
        published_explorer_match_frequency_hz = _nan_float_attr(
            root_attrs.get("published_explorer_match_frequency_hz")
        )
        published_explorer_hdf5_frequency_hz = _nan_float_attr(
            root_attrs.get("published_explorer_hdf5_frequency_hz")
        )
        published_explorer_label_frequency_hz = _nan_float_attr(
            root_attrs.get("published_explorer_label_frequency_hz")
        )
        published_explorer_match_max_abs_delta_db_1004_hz = _nan_float_attr(
            root_attrs.get("published_explorer_match_max_abs_delta_db_1004_hz")
        )
        published_explorer_match_max_abs_delta_angle_deg_1004_hz = _nan_float_attr(
            root_attrs.get("published_explorer_match_max_abs_delta_angle_deg_1004_hz")
        )
        driver_group = h5[driver_name]
        driver_attrs = driver_group.attrs
        group = driver_group[group_name]
        passive_state_status = _text_attr(driver_attrs.get("passive_state_status")) or _text_attr(
            group.attrs.get("passive_state_status")
        ) or ""
        passive_state_evidence = _text_attr(driver_attrs.get("passive_state_evidence")) or _text_attr(
            group.attrs.get("passive_state_evidence")
        ) or ""
        passive_state_acceptance_use = _text_attr(
            driver_attrs.get("passive_state_acceptance_use")
        ) or _text_attr(group.attrs.get("passive_state_acceptance_use")) or ""
        passive_state_metadata_policy = _text_attr(
            driver_attrs.get("passive_state_metadata_policy")
        ) or _text_attr(group.attrs.get("passive_state_metadata_policy")) or ""
        for angle_str in sorted(group.keys(), key=lambda value: int(value)):
            angle = int(angle_str)
            attrs = group[angle_str].attrs
            notes_by_angle[angle] = str(attrs.get("notes", ""))
            titles_by_angle[angle] = str(attrs.get("title", ""))
            timing_corrected_by_angle[angle] = _bool_attr(attrs.get("timing_corrected", False))
            timing_offset_ms_by_angle[angle] = float(attrs.get("timing_offset_ms", 0.0))
            timing_peak_time_ms_by_angle[angle] = _nan_float_attr(attrs.get("timing_peak_time_ms"))
            timing_global_peak_time_ms_by_angle[angle] = _nan_float_attr(
                attrs.get("timing_global_peak_time_ms")
            )
            timing_earliest_10pct_peak_time_ms_by_angle[angle] = _nan_float_attr(
                attrs.get("timing_earliest_10pct_peak_time_ms")
            )
            timing_first_strong_near_ref_lobe_time_ms_by_angle[angle] = _nan_float_attr(
                attrs.get("timing_first_strong_near_ref_lobe_time_ms")
            )
            timing_selected_minus_first_strong_near_ref_lobe_ms_by_angle[angle] = _nan_float_attr(
                attrs.get("timing_selected_minus_first_strong_near_ref_lobe_ms")
            )
            timing_selected_minus_first_strong_near_ref_lobe_path_mm_by_angle[angle] = (
                _nan_float_attr(attrs.get("timing_selected_minus_first_strong_near_ref_lobe_path_mm"))
            )
            if "timing_selected_is_first_strong_near_ref_lobe" in attrs:
                timing_selected_is_first_strong_near_ref_lobe_by_angle[angle] = _bool_attr(
                    attrs.get("timing_selected_is_first_strong_near_ref_lobe")
                )
            timing_mdat_window_ref_minus_first_strong_near_ref_lobe_path_mm_by_angle[angle] = (
                _nan_float_attr(attrs.get("timing_mdat_window_ref_minus_first_strong_near_ref_lobe_path_mm"))
            )
            if "timing_mdat_window_ref_is_first_strong_near_ref_lobe" in attrs:
                timing_mdat_window_ref_is_first_strong_near_ref_lobe_by_angle[angle] = _bool_attr(
                    attrs.get("timing_mdat_window_ref_is_first_strong_near_ref_lobe")
                )
            timing_mdat_window_ref_not_first_strong_near_ref_lobe_by_angle[angle] = _bool_attr(
                attrs.get("timing_mdat_window_ref_not_first_strong_near_ref_lobe", False)
            )
            timing_late_window_peak_time_ms_by_angle[angle] = _nan_float_attr(
                attrs.get("timing_late_window_peak_time_ms")
            )
            timing_late_window_peak_abs_rel_to_global_by_angle[angle] = _nan_float_attr(
                attrs.get("timing_late_window_peak_abs_rel_to_global")
            )
            timing_late_window_peak_warning_by_angle[angle] = _bool_attr(
                attrs.get("timing_late_window_peak_warning", False)
            )
            timing_peak_interpretation_by_angle[angle] = str(
                attrs.get("timing_peak_interpretation", "")
            )
            timing_peak_selection_reason_by_angle[angle] = str(
                attrs.get("timing_peak_selection_reason", "")
            )
            timing_peak_policy_by_angle[angle] = str(attrs.get("timing_peak_policy", ""))
            timing_current_loader_peak_rejected_by_angle[angle] = _bool_attr(
                attrs.get("timing_current_loader_peak_rejected", False)
            )
            timing_current_loader_selected_early_event_by_angle[angle] = _bool_attr(
                attrs.get("timing_current_loader_selected_early_event", False)
            )
            timing_suspicious_window_ref_alignment_by_angle[angle] = _bool_attr(
                attrs.get("timing_suspicious_window_ref_alignment", False)
            )
            timing_suspicious_reflection_alignment_by_angle[angle] = _bool_attr(
                attrs.get("timing_suspicious_reflection_alignment", False)
            )
            distance = _float_attr(attrs.get("measurement_distance_m"))
            height = _float_attr(attrs.get("measurement_height_m"))
            height_ref = _str_attr(attrs.get("measurement_height_reference"))
            if distance is not None:
                distance_attrs.append(distance)
            if height is not None:
                height_attrs.append(height)
            if height_ref:
                height_refs.append(height_ref)

    angles = tuple(sorted(notes_by_angle))
    all_text = "\n".join(notes_by_angle[angle] for angle in angles)
    return MeasurementNotes(
        hdf5_path=path,
        driver_name=driver_name,
        group_name=group_name,
        angles_deg=angles,
        notes_by_angle=notes_by_angle,
        titles_by_angle=titles_by_angle,
        timing_corrected_by_angle=timing_corrected_by_angle,
        timing_offset_ms_by_angle=timing_offset_ms_by_angle,
        timing_peak_time_ms_by_angle=timing_peak_time_ms_by_angle,
        timing_global_peak_time_ms_by_angle=timing_global_peak_time_ms_by_angle,
        timing_earliest_10pct_peak_time_ms_by_angle=timing_earliest_10pct_peak_time_ms_by_angle,
        timing_first_strong_near_ref_lobe_time_ms_by_angle=(
            timing_first_strong_near_ref_lobe_time_ms_by_angle
        ),
        timing_selected_minus_first_strong_near_ref_lobe_ms_by_angle=(
            timing_selected_minus_first_strong_near_ref_lobe_ms_by_angle
        ),
        timing_selected_minus_first_strong_near_ref_lobe_path_mm_by_angle=(
            timing_selected_minus_first_strong_near_ref_lobe_path_mm_by_angle
        ),
        timing_selected_is_first_strong_near_ref_lobe_by_angle=(
            timing_selected_is_first_strong_near_ref_lobe_by_angle
        ),
        timing_mdat_window_ref_minus_first_strong_near_ref_lobe_path_mm_by_angle=(
            timing_mdat_window_ref_minus_first_strong_near_ref_lobe_path_mm_by_angle
        ),
        timing_mdat_window_ref_is_first_strong_near_ref_lobe_by_angle=(
            timing_mdat_window_ref_is_first_strong_near_ref_lobe_by_angle
        ),
        timing_mdat_window_ref_not_first_strong_near_ref_lobe_by_angle=(
            timing_mdat_window_ref_not_first_strong_near_ref_lobe_by_angle
        ),
        timing_late_window_peak_time_ms_by_angle=timing_late_window_peak_time_ms_by_angle,
        timing_late_window_peak_abs_rel_to_global_by_angle=(
            timing_late_window_peak_abs_rel_to_global_by_angle
        ),
        timing_late_window_peak_warning_by_angle=timing_late_window_peak_warning_by_angle,
        timing_peak_interpretation_by_angle=timing_peak_interpretation_by_angle,
        timing_peak_selection_reason_by_angle=timing_peak_selection_reason_by_angle,
        timing_peak_policy_by_angle=timing_peak_policy_by_angle,
        timing_current_loader_peak_rejected_by_angle=timing_current_loader_peak_rejected_by_angle,
        timing_current_loader_selected_early_event_by_angle=(
            timing_current_loader_selected_early_event_by_angle
        ),
        timing_suspicious_window_ref_alignment_by_angle=timing_suspicious_window_ref_alignment_by_angle,
        timing_suspicious_reflection_alignment_by_angle=timing_suspicious_reflection_alignment_by_angle,
        target_kind=target_kind,
        diagnostic_only=diagnostic_only,
        not_acceptance_target=not_acceptance_target,
        validation_hypothesis=validation_hypothesis,
        processing_policy=processing_policy,
        peak_selection_policy=peak_selection_policy,
        gate_window_policy=gate_window_policy,
        normalization_policy=normalization_policy,
        published_polar_explorer_path=published_polar_explorer_path,
        published_polar_explorer_url=published_polar_explorer_url,
        published_explorer_match=published_explorer_match,
        published_explorer_match_frequency_hz=published_explorer_match_frequency_hz,
        published_explorer_hdf5_frequency_hz=published_explorer_hdf5_frequency_hz,
        published_explorer_label_frequency_hz=published_explorer_label_frequency_hz,
        published_explorer_match_max_abs_delta_db_1004_hz=(
            published_explorer_match_max_abs_delta_db_1004_hz
        ),
        published_explorer_match_max_abs_delta_angle_deg_1004_hz=(
            published_explorer_match_max_abs_delta_angle_deg_1004_hz
        ),
        parsed_distance_m=distance_attrs[0] if distance_attrs else parse_distance_m(all_text),
        parsed_height_m=height_attrs[0] if height_attrs else parse_height_m(all_text),
        parsed_height_reference=height_refs[0] if height_refs else parse_height_reference(all_text),
        passive_state_status=passive_state_status,
        passive_state_evidence=passive_state_evidence,
        passive_state_acceptance_use=passive_state_acceptance_use,
        passive_state_metadata_policy=passive_state_metadata_policy,
    )


def _float_attr(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _nan_float_attr(value: object) -> float:
    parsed = _float_attr(value)
    return float("nan") if parsed is None else parsed


def _bool_attr(value: object) -> bool:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _str_attr(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    parsed = str(value).strip().lower()
    return parsed or None


def _text_attr(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    parsed = str(value).strip()
    return parsed or None


def parse_distance_m(text: str) -> float | None:
    """Parse measurement distance from common Spanish/REW note fragments."""

    candidates = []
    for match in re.finditer(r"(?<![-\d.])(\d+(?:[.,]\d+)?)\s*(?:m|metros?)\b", text, re.IGNORECASE):
        value = float(match.group(1).replace(",", "."))
        if 0.05 <= value <= 20.0:
            candidates.append(value)
    for match in re.finditer(r"(?<![-\d.])(\d+(?:[.,]\d+)?)\s*(?:cm|cms|centimetros?)\b", text, re.IGNORECASE):
        value = float(match.group(1).replace(",", ".")) / 100.0
        if 0.05 <= value <= 20.0:
            candidates.append(value)
    return candidates[0] if candidates else None


def parse_delay_note_ms(text: str) -> float | None:
    match = re.search(r"Delay\s+([-+0-9.,]+)\s+ms", text, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def parse_height_m(text: str) -> float | None:
    for line in text.lower().splitlines():
        for match in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(?:cm|cms)\s*(?:de\s*)?(?:altura|alto)", line):
            return float(match.group(1).replace(",", ".")) / 100.0
        for match in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(?:de\s*)?(?:altura|alto)", line):
            value = float(match.group(1).replace(",", "."))
            if value > 20.0:
                return value / 100.0
            if 0.3 <= value <= 3.0:
                return value
        for match in re.finditer(r"(?:altura|alto)\D{0,12}(\d+(?:[.,]\d+)?)", line):
            value = float(match.group(1).replace(",", "."))
            if value > 20.0:
                return value / 100.0
            if 0.3 <= value <= 3.0:
                return value
    return None


def parse_height_reference(text: str) -> str | None:
    """Parse named microphone-height references such as 'altura UM' or 'Mic height: L22MG/LM'."""

    lowered = text.lower()
    if re.search(r"\baltura\s*(?:de\s*)?u\.?m\.?\b", lowered) or re.search(
        r"\bum\s*(?:height|altura)\b",
        lowered,
    ):
        return "um"
    if re.search(r"\bmic\s*height\s*:\s*(?:l22mg|l22|lm)\b", lowered) or re.search(
        r"\b(?:l22mg|l22)\s*/\s*lm\b",
        lowered,
    ):
        return "l22mg"
    return None
