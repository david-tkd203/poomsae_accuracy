#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend real para Kinect v2 usando PyKinect2
Detecta y activa sensores RGB, Depth, IR y Audio
"""
try:
	from pykinect2 import PyKinectRuntime, PyKinectV2
	import numpy as np
	import cv2
	KINECT_AVAILABLE = True
except ImportError:
	KINECT_AVAILABLE = False


# --- NUEVO BACKEND AVANZADO ---
import math
class KinectPoomsaeCapture:
	"""Backend avanzado para Kinect v2 real con PyKinect2 y análisis biomecánico completo"""
	def __init__(self, enable_color=True, enable_depth=True, enable_ir=True, enable_audio=True, enable_body=True):
		self.enable_color = enable_color
		self.enable_depth = enable_depth
		self.enable_ir = enable_ir
		self.enable_audio = enable_audio
		self.enable_body = enable_body
		self.kinect = None
		self.sensors = {}
		self.frame_count = 0

	def initialize(self):
		if not KINECT_AVAILABLE:
			print("PyKinect2 no disponible o Kinect no conectado.")
			return False
		try:
			sources = 0
			if self.enable_color:
				sources |= PyKinectV2.FrameSourceTypes_Color
			if self.enable_depth:
				sources |= PyKinectV2.FrameSourceTypes_Depth
			if self.enable_ir:
				sources |= PyKinectV2.FrameSourceTypes_Infrared
			if self.enable_audio:
				sources |= PyKinectV2.FrameSourceTypes_Audio
			if self.enable_body:
				sources |= PyKinectV2.FrameSourceTypes_Body
			self.kinect = PyKinectRuntime.PyKinectRuntime(sources)
			self.sensors = {
				'color': self.enable_color and self.kinect.color_frame_desc is not None,
				'depth': self.enable_depth and self.kinect.depth_frame_desc is not None,
				'ir': self.enable_ir and hasattr(self.kinect, 'infrared_frame_desc'),
				'audio': self.enable_audio and hasattr(self.kinect, 'audio_frame_desc'),
				'body': self.enable_body
			}
			print(f"Sensores activos: {', '.join([k for k,v in self.sensors.items() if v])}")
			return True
		except Exception as e:
			print(f"Error inicializando Kinect: {e}")
			return False

	def get_frame(self):
		if not self.kinect:
			return None
		frame_data = {}
		self.frame_count += 1
		# Color
		if self.sensors.get('color') and self.kinect.has_new_color_frame():
			color_frame = self.kinect.get_last_color_frame()
			color_img = color_frame.reshape((1080, 1920, 4)).astype(np.uint8)[..., :3]
			frame_data['rgb'] = color_img
		# Depth
		if self.sensors.get('depth') and self.kinect.has_new_depth_frame():
			depth_frame = self.kinect.get_last_depth_frame()
			depth_img = depth_frame.reshape((424, 512)).astype(np.uint16)
			frame_data['depth'] = depth_img
		# Infrared
		if self.sensors.get('ir') and self.kinect.has_new_infrared_frame():
			ir_frame = self.kinect.get_last_infrared_frame()
			ir_img = ir_frame.reshape((424, 512)).astype(np.uint16)
			frame_data['ir'] = ir_img
		# Audio (solo referencia, no procesamiento en este backend)
		if self.sensors.get('audio'):
			frame_data['audio'] = 'activo'
		# Body tracking y análisis biomecánico
		frame_data['bodies'] = []
		frame_data['biomechanics'] = []
		if self.sensors.get('body') and self.kinect.has_new_body_frame():
			bodies = self.kinect.get_last_body_frame()
			if bodies is not None:
				for i in range(0, self.kinect.max_body_count):
					body = bodies.bodies[i]
					if not body.is_tracked:
						continue
					joints = body.joints
					landmarks = {}
					for j in joints:
						pos = joints[j].Position
						landmarks[j] = (pos.x, pos.y, pos.z, 1.0)
					frame_data['bodies'].append(landmarks)
					# --- Análisis biomecánico específico para poomsae Pal Yang ---
					angles = self._calculate_angles(landmarks)
					metrics = self._biomechanics_metrics(landmarks, angles)
					frame_data['biomechanics'].append({
						'angles': angles,
						'metrics': metrics
					})
		# Timestamp robusto según sensor disponible
		ts = None
		if self.sensors.get('color') and hasattr(self.kinect, '_last_color_frame_time'):
			ts = self.kinect._last_color_frame_time
		elif self.sensors.get('depth') and hasattr(self.kinect, '_last_depth_frame_time'):
			ts = self.kinect._last_depth_frame_time
		elif self.sensors.get('body') and hasattr(self.kinect, '_last_body_frame_time'):
			ts = self.kinect._last_body_frame_time
		frame_data['timestamp'] = ts if ts is not None else self.frame_count
		frame_data['frame_number'] = self.frame_count
		return frame_data if frame_data else None

	def _calculate_angles(self, landmarks):
		# Ángulos articulares clave para taekwondo (Pal Yang)
		# Basado en los joints de PyKinectV2
		def angle3d(a, b, c):
			try:
				ax, ay, az = a[0], a[1], a[2]
				bx, by, bz = b[0], b[1], b[2]
				cx, cy, cz = c[0], c[1], c[2]
				ba = [ax-bx, ay-by, az-bz]
				bc = [cx-bx, cy-by, cz-bz]
				dot = sum([ba[i]*bc[i] for i in range(3)])
				norm_ba = math.sqrt(sum([ba[i]**2 for i in range(3)]))
				norm_bc = math.sqrt(sum([bc[i]**2 for i in range(3)]))
				if norm_ba == 0 or norm_bc == 0:
					return None
				cos_angle = dot / (norm_ba * norm_bc)
				angle = math.acos(max(-1.0, min(1.0, cos_angle)))
				return math.degrees(angle)
			except:
				return None
		# Joints relevantes
		J = PyKinectV2.JointType
		angles = {}
		# Codo
		for side in ['Left', 'Right']:
			try:
				angles[f'{side.lower()}_elbow'] = angle3d(
					landmarks[getattr(J, f'{side}Shoulder')],
					landmarks[getattr(J, f'{side}Elbow')],
					landmarks[getattr(J, f'{side}Wrist')]
				)
			except:
				pass
		# Rodilla
		for side in ['Left', 'Right']:
			try:
				angles[f'{side.lower()}_knee'] = angle3d(
					landmarks[getattr(J, f'{side}Hip')],
					landmarks[getattr(J, f'{side}Knee')],
					landmarks[getattr(J, f'{side}Ankle')]
				)
			except:
				pass
		# Hombro
		for side in ['Left', 'Right']:
			try:
				angles[f'{side.lower()}_shoulder'] = angle3d(
					landmarks[getattr(J, f'{side}Hip')],
					landmarks[getattr(J, f'{side}Shoulder')],
					landmarks[getattr(J, f'{side}Elbow')]
				)
			except:
				pass
		# Cadera
		for side in ['Left', 'Right']:
			try:
				angles[f'{side.lower()}_hip'] = angle3d(
					landmarks[getattr(J, f'{side}Shoulder')],
					landmarks[getattr(J, f'{side}Hip')],
					landmarks[getattr(J, f'{side}Knee')]
				)
			except:
				pass
		return angles

	def _biomechanics_metrics(self, landmarks, angles):
		# Métricas biomecánicas para evaluación de poomsae
		metrics = {}
		# Desbalance: diferencia de altura entre hombros/caderas
		try:
			l_sh = landmarks[PyKinectV2.JointType_ShoulderLeft][1]
			r_sh = landmarks[PyKinectV2.JointType_ShoulderRight][1]
			l_hip = landmarks[PyKinectV2.JointType_HipLeft][1]
			r_hip = landmarks[PyKinectV2.JointType_HipRight][1]
			metrics['shoulder_balance'] = abs(l_sh - r_sh)
			metrics['hip_balance'] = abs(l_hip - r_hip)
		except:
			metrics['shoulder_balance'] = None
			metrics['hip_balance'] = None
		# Postura: ángulo de torso
		try:
			l_sh = landmarks[PyKinectV2.JointType_ShoulderLeft]
			r_sh = landmarks[PyKinectV2.JointType_ShoulderRight]
			l_hip = landmarks[PyKinectV2.JointType_HipLeft]
			r_hip = landmarks[PyKinectV2.JointType_HipRight]
			torso_angle = self._calculate_angles({
				'a': l_sh, 'b': l_hip, 'c': r_hip
			})
			metrics['torso_angle'] = torso_angle.get('a_elbow', None)
		except:
			metrics['torso_angle'] = None
		# Extensión: codo y rodilla
		for k in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']:
			metrics[f'extension_{k}'] = angles.get(k, None)
		# Grave: rango de cadera y rodilla
		for k in ['left_hip', 'right_hip', 'left_knee', 'right_knee']:
			ang = angles.get(k, None)
			if ang is not None:
				metrics[f'grave_{k}'] = ang < 120 or ang > 180
			else:
				metrics[f'grave_{k}'] = None
		return metrics

	def visualize_frame(self, frame_data):
		# Visualización avanzada de todos los sensores y métricas
		import cv2, numpy as np
		rgb_frame = frame_data.get('rgb')
		depth_frame = frame_data.get('depth')
		ir_frame = frame_data.get('ir')
		vis = None
		if rgb_frame is not None:
			vis = rgb_frame.copy()
			h, w = vis.shape[:2]
			# Overlay de depth
			if depth_frame is not None:
				depth_colormap = cv2.applyColorMap(
					cv2.convertScaleAbs(depth_frame, alpha=0.08), cv2.COLORMAP_JET
				)
				if depth_colormap.shape[:2] != vis.shape[:2]:
					depth_colormap = cv2.resize(depth_colormap, (w, h))
				vis = np.hstack((vis, depth_colormap))
			# Overlay de IR
			if ir_frame is not None:
				ir_colormap = cv2.applyColorMap(
					cv2.convertScaleAbs(ir_frame, alpha=0.08), cv2.COLORMAP_BONE
				)
				if ir_colormap.shape[:2] != vis.shape[:2]:
					ir_colormap = cv2.resize(ir_colormap, (w, h))
				vis = np.hstack((vis, ir_colormap))
			# Overlay de cuerpos y métricas
			if 'bodies' in frame_data:
				for idx, landmarks in enumerate(frame_data['bodies']):
					for lm_id, (x, y, z, visib) in landmarks.items():
						px, py = int(x * w), int(y * h)
						cv2.circle(vis, (px, py), 4, (0,255,0), -1)
					# Overlay de métricas biomecánicas
					if 'biomechanics' in frame_data and idx < len(frame_data['biomechanics']):
						metrics = frame_data['biomechanics'][idx]['metrics']
						y_pos = 30 + idx*120
						for k, v in metrics.items():
							cv2.putText(vis, f"{k}: {v}", (10, y_pos),
									   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
							y_pos += 18
		elif depth_frame is not None:
			vis = cv2.applyColorMap(
				cv2.convertScaleAbs(depth_frame, alpha=0.08), cv2.COLORMAP_JET
			)
		elif ir_frame is not None:
			vis = cv2.applyColorMap(
				cv2.convertScaleAbs(ir_frame, alpha=0.08), cv2.COLORMAP_BONE
			)
		else:
			vis = np.zeros((480, 640, 3), dtype=np.uint8)
		# Info overlay
		info = [
			f"Kinect Real - Frame {frame_data.get('frame_number', 0)}",
			f"Sensores: {', '.join([k for k,v in self.sensors.items() if v])}"
		]
		y_pos = 25
		for text in info:
			cv2.putText(vis, text, (10, y_pos),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
			y_pos += 25
		return vis

	def cleanup(self):
		if self.kinect:
			self.kinect.close()
			print("Kinect liberada.")

def detect_kinect_device():
	return KINECT_AVAILABLE
