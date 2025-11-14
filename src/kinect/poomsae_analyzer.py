#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de análisis y reporte para poomsae
"""
import json, csv
from pathlib import Path

class PoomsaeReport:
	def __init__(self, poomsae_name="8yang"):
		self.poomsae_name = poomsae_name
		self.frames = []

	def add_frame(self, frame_number, timestamp, discounts, errors, angles, pose_confidence):
		self.frames.append({
			'frame': frame_number,
			'timestamp': timestamp,
			'discounts': discounts,
			'errors': errors,
			'angles': angles,
			'pose_confidence': pose_confidence
		})

	def save_json(self, out_path):
		with open(out_path, 'w') as f:
			json.dump(self.frames, f, indent=2)
		print(f"✅ Reporte JSON guardado: {out_path}")

	def save_csv(self, out_path):
		if not self.frames:
			print("No hay datos para exportar.")
			return
		keys = ['frame', 'timestamp', 'discounts', 'errors', 'pose_confidence'] + [f"angle_{k}" for k in self.frames[0]['angles'].keys()]
		with open(out_path, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=keys)
			writer.writeheader()
			for fr in self.frames:
				row = {k: fr.get(k, '') for k in ['frame', 'timestamp', 'discounts', 'errors', 'pose_confidence']}
				for ak, av in fr['angles'].items():
					row[f"angle_{ak}"] = av
				writer.writerow(row)
		print(f"✅ Reporte CSV guardado: {out_path}")

	def summary(self):
		total_discount = sum(sum(f['discounts']) for f in self.frames)
		print(f"Resumen: {len(self.frames)} frames, descuento total acumulado: {total_discount:.2f}")
