# src/eval/spec_validator.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class MoveSpec:
    seq: int
    idx: int
    sub: Optional[str]
    tech_kor: str
    tech_es: str
    stance_code: str
    level: str
    lead: str
    turn: str
    travel: str
    category: str
    active_limb: str
    side_inferred: str
    stance_expect: str
    turn_expect: str
    kick_type: Optional[str]
    timing: Dict[str, float]

@dataclass
class PoomsaeSpec:
    poomsae: str
    steps_total: int
    segments_total: int
    moves: List[MoveSpec]
    order: List[str]
    scoring: Dict[str, Any]
    
    @classmethod
    def from_file(cls, file_path: Path) -> PoomsaeSpec:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        moves = []
        for move_data in data['moves']:
            moves.append(MoveSpec(
                seq=move_data['seq'],
                idx=move_data['idx'],
                sub=move_data.get('sub'),
                tech_kor=move_data['tech_kor'],
                tech_es=move_data['tech_es'],
                stance_code=move_data['stance_code'],
                level=move_data['level'],
                lead=move_data['lead'],
                turn=move_data['turn'],
                travel=move_data['travel'],
                category=move_data['category'],
                active_limb=move_data['active_limb'],
                side_inferred=move_data['side_inferred'],
                stance_expect=move_data['stance_expect'],
                turn_expect=move_data['turn_expect'],
                kick_type=move_data.get('kick_type'),
                timing=move_data['timing']
            ))
        
        return cls(
            poomsae=data['poomsae'],
            steps_total=data['steps_total'],
            segments_total=data['segments_total'],
            moves=moves,
            order=data['order'],
            scoring=data['scoring']
        )

@dataclass
class PoseSpec:
    tolerances: Dict[str, float]
    levels: Dict[str, Any]
    stances: Dict[str, Any]
    kicks: Dict[str, Any]
    
    @classmethod
    def from_file(cls, file_path: Path) -> PoseSpec:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            tolerances=data['tolerances'],
            levels=data['levels'],
            stances=data['stances'],
            kicks=data['kicks']
        )

class SpecValidator:
    def __init__(self, poomsae_spec_path: Path, pose_spec_path: Path):
        self.poomsae_spec = PoomsaeSpec.from_file(poomsae_spec_path)
        self.pose_spec = PoseSpec.from_file(pose_spec_path)
    
    def validate_move(self, move_idx: int, detected_move: Dict[str, Any], 
                     landmarks: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Valida un movimiento detectado contra la especificación"""
        if move_idx >= len(self.poomsae_spec.moves):
            return {"valid": False, "error": "Move index out of range"}
        
        expected = self.poomsae_spec.moves[move_idx]
        results = {
            "valid": True,
            "deductions": 0.0,
            "errors": [],
            "warnings": []
        }
        
        # Validar extremidad activa
        if not self._validate_active_limb(detected_move['active_limb'], expected.active_limb):
            results["errors"].append(f"Extremidad activa incorrecta: {detected_move['active_limb']} vs {expected.active_limb}")
            results["deductions"] += 0.1
        
        # Validar postura
        stance_result = self._validate_stance(detected_move['stance_pred'], expected.stance_expect, landmarks)
        if not stance_result["valid"]:
            results["errors"].extend(stance_result["errors"])
            results["deductions"] += stance_result["deduction"]
        
        # Validar rotación
        rotation_result = self._validate_rotation(detected_move['rotation_bucket'], expected.turn_expect)
        if not rotation_result["valid"]:
            results["errors"].extend(rotation_result["errors"])
            results["deductions"] += rotation_result["deduction"]
        
        # Validar patada
        if expected.kick_type:
            kick_result = self._validate_kick(detected_move['kick_pred'], expected.kick_type, landmarks)
            if not kick_result["valid"]:
                results["errors"].extend(kick_result["errors"])
                results["deductions"] += kick_result["deduction"]
        
        # Validar timing
        timing_result = self._validate_timing(detected_move['duration'], expected.timing)
        if not timing_result["valid"]:
            results["warnings"].extend(timing_result["warnings"])
        
        if results["deductions"] > 0:
            results["valid"] = False
        
        return results
    
    def _validate_active_limb(self, detected: str, expected: str) -> bool:
        return detected == expected
    
    def _validate_stance(self, detected: str, expected: str, landmarks: Dict[str, np.ndarray]) -> Dict[str, Any]:
        result = {"valid": True, "errors": [], "deduction": 0.0}
        
        if detected != expected:
            result["valid"] = False
            result["errors"].append(f"Postura incorrecta: {detected} vs {expected}")
            result["deduction"] = 0.1
        
        # Validación adicional con pose_spec
        if expected in self.pose_spec.stances:
            stance_spec = self.pose_spec.stances[expected]
            # Aquí agregar validaciones específicas de la postura
            # usando los umbrales de pose_spec.json
        
        return result
    
    def _validate_rotation(self, detected: str, expected: str) -> Dict[str, Any]:
        result = {"valid": True, "errors": [], "deduction": 0.0}
        
        if detected != expected and expected != "NONE":
            result["valid"] = False
            result["errors"].append(f"Rotación incorrecta: {detected} vs {expected}")
            result["deduction"] = 0.3  # Error grave por rotación incorrecta
        
        return result
    
    def _validate_kick(self, detected: str, expected: str, landmarks: Dict[str, np.ndarray]) -> Dict[str, Any]:
        result = {"valid": True, "errors": [], "deduction": 0.0}
        
        if detected != expected:
            result["valid"] = False
            result["errors"].append(f"Tipo de patada incorrecto: {detected} vs {expected}")
            result["deduction"] = 0.1
        
        # Validación adicional con pose_spec
        if expected in self.pose_spec.kicks:
            kick_spec = self.pose_spec.kicks[expected]
            # Aquí agregar validaciones específicas de la patada
        
        return result
    
    def _validate_timing(self, duration: float, timing_spec: Dict[str, float]) -> Dict[str, Any]:
        result = {"valid": True, "warnings": []}
        
        min_time = timing_spec.get("min_s", 0.1)
        max_time = timing_spec.get("max_s", 2.0)
        
        if duration < min_time:
            result["warnings"].append(f"Duración muy corta: {duration:.2f}s < {min_time:.2f}s")
        elif duration > max_time:
            result["warnings"].append(f"Duración muy larga: {duration:.2f}s > {max_time:.2f}s")
        
        return result
    
    def get_expected_move(self, move_idx: int) -> MoveSpec:
        """Obtiene la especificación esperada para un movimiento"""
        return self.poomsae_spec.moves[move_idx]
    
    def get_total_moves(self) -> int:
        """Retorna el número total de movimientos esperados"""
        return self.poomsae_spec.segments_total