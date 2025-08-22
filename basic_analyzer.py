import cv2
import numpy as np
import mediapipe as mp
import json
import time
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BiomechanicalMetrics:
    """Store biomechanical measurements for a single frame"""
    frame_number: int
    timestamp: float
    front_elbow_angle: Optional[float] = None
    spine_lean: Optional[float] = None
    head_knee_alignment: Optional[float] = None
    front_foot_direction: Optional[float] = None
    pose_confidence: float = 0.0
    
@dataclass
class ShotEvaluation:
    """Final shot evaluation scores and feedback"""
    footwork_score: float
    head_position_score: float
    swing_control_score: float
    balance_score: float
    follow_through_score: float
    overall_score: float
    footwork_feedback: str
    head_position_feedback: str
    swing_control_feedback: str
    balance_feedback: str
    follow_through_feedback: str

class PoseEstimator:
    """Handles pose estimation using MediaPipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose model with optimized settings for speed
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balance between accuracy and speed
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_keypoints(self, frame: np.ndarray) -> Tuple[Optional[Any], float]:
        """Extract pose keypoints from frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Calculate average visibility as confidence metric
                confidence = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])
                return results, confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.warning(f"Pose estimation failed: {e}")
            return None, 0.0
    
    def get_landmark_coords(self, landmarks, landmark_id: int, frame_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """Convert normalized landmarks to pixel coordinates"""
        try:
            if landmarks and len(landmarks.landmark) > landmark_id:
                lm = landmarks.landmark[landmark_id]
                if lm.visibility > 0.5:
                    h, w = frame_shape[:2]
                    return (int(lm.x * w), int(lm.y * h))
            return None
        except:
            return None

class BiomechanicsAnalyzer:
    """Analyzes biomechanical metrics from pose data"""
    
    def __init__(self):
        # Define MediaPipe landmark indices
        self.LANDMARK_IDS = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        # Smoothing buffers for temporal consistency
        self.angle_buffer_size = 5
        self.elbow_angles = deque(maxlen=self.angle_buffer_size)
        self.spine_leans = deque(maxlen=self.angle_buffer_size)
        
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> Optional[float]:
        """Calculate angle at point p2 formed by p1-p2-p3"""
        try:
            if None in [p1, p2, p3]:
                return None
                
            # Convert to numpy arrays
            a = np.array(p1)
            b = np.array(p2)
            c = np.array(p3)
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Handle numerical errors
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
            
        except Exception as e:
            logger.debug(f"Angle calculation failed: {e}")
            return None
    
    def calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Optional[float]:
        """Calculate Euclidean distance between two points"""
        try:
            if None in [p1, p2]:
                return None
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        except:
            return None
    
    def analyze_frame(self, landmarks, frame_shape: Tuple[int, int], frame_number: int, timestamp: float) -> BiomechanicalMetrics:
        """Analyze biomechanics for a single frame"""
        
        pose_estimator = PoseEstimator()
        
        # Extract key landmarks
        nose = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['nose'], frame_shape)
        left_shoulder = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['left_shoulder'], frame_shape)
        right_shoulder = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['right_shoulder'], frame_shape)
        left_elbow = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['left_elbow'], frame_shape)
        right_elbow = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['right_elbow'], frame_shape)
        left_wrist = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['left_wrist'], frame_shape)
        right_wrist = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['right_wrist'], frame_shape)
        left_hip = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['left_hip'], frame_shape)
        right_hip = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['right_hip'], frame_shape)
        left_knee = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['left_knee'], frame_shape)
        right_knee = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['right_knee'], frame_shape)
        
        # Initialize metrics
        metrics = BiomechanicalMetrics(frame_number=frame_number, timestamp=timestamp)
        
        # 1. Front elbow angle (assuming right-handed batsman, left elbow is front)
        front_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        if front_elbow_angle:
            self.elbow_angles.append(front_elbow_angle)
            # Use smoothed angle
            metrics.front_elbow_angle = np.mean(list(self.elbow_angles))
        
        # 2. Spine lean (hip-shoulder line vs vertical)
        if left_shoulder and right_shoulder and left_hip and right_hip:
            # Calculate shoulder center and hip center
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                             (left_shoulder[1] + right_shoulder[1]) / 2)
            hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2)
            
            # Calculate spine angle relative to vertical
            spine_vector = (shoulder_center[0] - hip_center[0], shoulder_center[1] - hip_center[1])
            vertical_vector = (0, -1)  # Pointing up
            
            if spine_vector[1] != 0:  # Avoid division by zero
                spine_angle = math.degrees(math.atan2(spine_vector[0], -spine_vector[1]))
                self.spine_leans.append(abs(spine_angle))
                metrics.spine_lean = np.mean(list(self.spine_leans))
        
        # 3. Head-over-knee alignment (vertical distance)
        if nose and left_knee and right_knee:
            # Use front knee (left for right-handed batsman)
            front_knee = left_knee
            metrics.head_knee_alignment = abs(nose[0] - front_knee[0])  # Horizontal offset
        
        # 4. Front foot direction (simplified as ankle-knee angle)
        front_ankle = pose_estimator.get_landmark_coords(landmarks, self.LANDMARK_IDS['left_ankle'], frame_shape)
        if left_knee and front_ankle:
            # Calculate foot direction relative to horizontal
            foot_vector = (front_ankle[0] - left_knee[0], front_ankle[1] - left_knee[1])
            horizontal_vector = (1, 0)
            
            if foot_vector[0] != 0:
                foot_angle = math.degrees(math.atan2(foot_vector[1], foot_vector[0]))
                metrics.front_foot_direction = abs(foot_angle)
        
        # Calculate pose confidence
        if landmarks:
            visible_landmarks = [lm for lm in landmarks.landmark if lm.visibility > 0.5]
            metrics.pose_confidence = len(visible_landmarks) / len(landmarks.landmark)
        
        return metrics

class VideoOverlayRenderer:
    """Handles video annotation and overlay rendering"""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 1
        
        # Color scheme
        self.colors = {
            'good': (0, 255, 0),      # Green
            'warning': (0, 255, 255),  # Yellow
            'poor': (0, 0, 255),      # Red
            'text': (255, 255, 255),   # White
            'skeleton': (0, 255, 255)  # Yellow
        }
    
    def draw_skeleton(self, frame: np.ndarray, landmarks, pose_estimator: PoseEstimator) -> np.ndarray:
        """Draw pose skeleton on frame"""
        if landmarks:
            # Draw landmarks and connections
            pose_estimator.mp_drawing.draw_landmarks(
                frame, 
                landmarks,
                pose_estimator.mp_pose.POSE_CONNECTIONS,
                pose_estimator.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def get_feedback_color(self, value: Optional[float], good_range: Tuple[float, float], 
                          warning_range: Tuple[float, float]) -> Tuple[Tuple[int, int, int], str]:
        """Determine feedback color and message based on value ranges"""
        if value is None:
            return self.colors['text'], "N/A"
        
        if good_range[0] <= value <= good_range[1]:
            return self.colors['good']
        elif warning_range[0] <= value <= warning_range[1]:
            return self.colors['warning']
        else:
            return self.colors['poor']
    
    def draw_metrics_overlay(self, frame: np.ndarray, metrics: BiomechanicalMetrics) -> np.ndarray:
        """Draw real-time metrics overlay on frame"""
        h, w = frame.shape[:2]
        y_offset = 30
        
        # Background panel for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (180, 125), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Elbow angle feedback
        if metrics.front_elbow_angle is not None:
            color = self.get_feedback_color(metrics.front_elbow_angle, (100, 140), (80, 160))
            text = f"Front Elbow: {metrics.front_elbow_angle:.1f}°"
            cv2.putText(frame, text, (20, y_offset), self.font, self.font_scale, color, self.thickness)
            y_offset += 25
        
        # Spine lean feedback
        if metrics.spine_lean is not None:
            color = self.get_feedback_color(metrics.spine_lean, (0, 15), (15, 25))
            text = f"Spine Lean: {metrics.spine_lean:.1f}°"
            cv2.putText(frame, text, (20, y_offset), self.font, self.font_scale, color, self.thickness)
            y_offset += 25
        
        # Head-knee alignment feedback
        if metrics.head_knee_alignment is not None:
            # Normalize by frame width for consistent feedback
            normalized_alignment = metrics.head_knee_alignment / w * 100
            color = self.get_feedback_color(normalized_alignment, (0, 10), (10, 20))
            text = f"Head-Knee: {normalized_alignment:.1f}%"
            cv2.putText(frame, text, (20, y_offset), self.font, self.font_scale, color, self.thickness)
            y_offset += 25
        
        # Foot direction feedback
        if metrics.front_foot_direction is not None:
            color = self.get_feedback_color(metrics.front_foot_direction, (0, 30), (30, 45))
            text = f"Foot Angle: {metrics.front_foot_direction:.1f}°"
            cv2.putText(frame, text, (20, y_offset), self.font, self.font_scale, color, self.thickness)
            y_offset += 25
        
        # Pose confidence
        conf_color = self.colors['good'] if metrics.pose_confidence > 0.8 else self.colors['warning']
        text = f"Pose Conf: {metrics.pose_confidence:.2f}"
        cv2.putText(frame, text, (20, y_offset), self.font, self.font_scale, conf_color, self.thickness)
        
        return frame

class ShotEvaluator:
    """Evaluates overall shot quality and provides feedback"""
    
    def __init__(self):
        self.metrics_history: List[BiomechanicalMetrics] = []
    
    def add_metrics(self, metrics: BiomechanicalMetrics):
        """Add frame metrics to history"""
        self.metrics_history.append(metrics)
    
    def evaluate_shot(self) -> ShotEvaluation:
        """Generate final shot evaluation"""
        if not self.metrics_history:
            return self._default_evaluation()
        
        # Filter out None values and calculate statistics
        elbow_angles = [m.front_elbow_angle for m in self.metrics_history if m.front_elbow_angle is not None]
        spine_leans = [m.spine_lean for m in self.metrics_history if m.spine_lean is not None]
        alignments = [m.head_knee_alignment for m in self.metrics_history if m.head_knee_alignment is not None]
        foot_angles = [m.front_foot_direction for m in self.metrics_history if m.front_foot_direction is not None]
        confidences = [m.pose_confidence for m in self.metrics_history]
        
        # Calculate scores (1-10 scale)
        footwork_score = self._evaluate_footwork(foot_angles)
        head_position_score = self._evaluate_head_position(alignments)
        swing_control_score = self._evaluate_swing_control(elbow_angles)
        balance_score = self._evaluate_balance(spine_leans)
        follow_through_score = self._evaluate_follow_through(elbow_angles, spine_leans)
        
        # Overall score (weighted average)
        overall_score = (
            footwork_score * 0.2 +
            head_position_score * 0.25 +
            swing_control_score * 0.25 +
            balance_score * 0.15 +
            follow_through_score * 0.15
        )
        
        return ShotEvaluation(
            footwork_score=footwork_score,
            head_position_score=head_position_score,
            swing_control_score=swing_control_score,
            balance_score=balance_score,
            follow_through_score=follow_through_score,
            overall_score=overall_score,
            footwork_feedback=self._get_footwork_feedback(footwork_score, foot_angles),
            head_position_feedback=self._get_head_position_feedback(head_position_score, alignments),
            swing_control_feedback=self._get_swing_control_feedback(swing_control_score, elbow_angles),
            balance_feedback=self._get_balance_feedback(balance_score, spine_leans),
            follow_through_feedback=self._get_follow_through_feedback(follow_through_score)
        )
    
    def _evaluate_footwork(self, foot_angles: List[float]) -> float:
        """Evaluate footwork quality (1-10)"""
        if not foot_angles:
            return 5.0  # Neutral score when no data
        
        avg_angle = np.mean(foot_angles)
        consistency = 10 - np.std(foot_angles) * 0.2  # Penalize inconsistency
        
        # Ideal foot angle is around 15-30 degrees
        if 15 <= avg_angle <= 30:
            angle_score = 10
        elif 10 <= avg_angle <= 40:
            angle_score = 8
        else:
            angle_score = 5
        
        return max(1, min(10, (angle_score + consistency) / 2))
    
    def _evaluate_head_position(self, alignments: List[float]) -> float:
        """Evaluate head position quality (1-10)"""
        if not alignments:
            return 5.0
        
        # Smaller alignment values are better (head closer to over front knee)
        avg_alignment = np.mean(alignments)
        consistency = 10 - np.std(alignments) * 0.1
        
        # Score based on average alignment (normalized)
        if avg_alignment < 20:  # Very good alignment
            alignment_score = 10
        elif avg_alignment < 50:
            alignment_score = 8
        elif avg_alignment < 100:
            alignment_score = 6
        else:
            alignment_score = 4
        
        return max(1, min(10, (alignment_score + consistency) / 2))
    
    def _evaluate_swing_control(self, elbow_angles: List[float]) -> float:
        """Evaluate swing control quality (1-10)"""
        if not elbow_angles:
            return 5.0
        
        avg_angle = np.mean(elbow_angles)
        consistency = 10 - np.std(elbow_angles) * 0.1
        
        # Ideal elbow angle is around 110-130 degrees
        if 110 <= avg_angle <= 130:
            angle_score = 10
        elif 100 <= avg_angle <= 140:
            angle_score = 8
        elif 90 <= avg_angle <= 150:
            angle_score = 6
        else:
            angle_score = 4
        
        return max(1, min(10, (angle_score + consistency) / 2))
    
    def _evaluate_balance(self, spine_leans: List[float]) -> float:
        """Evaluate balance quality (1-10)"""
        if not spine_leans:
            return 5.0
        
        avg_lean = np.mean(spine_leans)
        consistency = 10 - np.std(spine_leans) * 0.2
        
        # Less spine lean is generally better for balance
        if avg_lean < 10:
            lean_score = 10
        elif avg_lean < 20:
            lean_score = 8
        elif avg_lean < 30:
            lean_score = 6
        else:
            lean_score = 4
        
        return max(1, min(10, (lean_score + consistency) / 2))
    
    def _evaluate_follow_through(self, elbow_angles: List[float], spine_leans: List[float]) -> float:
        """Evaluate follow-through quality (1-10)"""
        # This is a simplified evaluation - could be improved with phase detection
        if not elbow_angles or not spine_leans:
            return 5.0
        
        # Look for progression in angles (indicating follow-through)
        if len(elbow_angles) > 10:
            early_angles = np.mean(elbow_angles[:len(elbow_angles)//3])
            late_angles = np.mean(elbow_angles[-len(elbow_angles)//3:])
            angle_progression = abs(late_angles - early_angles)
            
            if angle_progression > 20:  # Good follow-through movement
                return 8.0
            elif angle_progression > 10:
                return 6.0
            else:
                return 4.0
        
        return 5.0
    
    def _get_footwork_feedback(self, score: float, foot_angles: List[float]) -> str:
        """Generate footwork feedback"""
        if score >= 8:
            return "Excellent foot positioning and movement. Good stride toward the pitch."
        elif score >= 6:
            return "Good footwork. Consider more decisive movement toward the ball."
        else:
            return "Work on foot positioning. Plant front foot closer to pitch line."
    
    def _get_head_position_feedback(self, score: float, alignments: List[float]) -> str:
        """Generate head position feedback"""
        if score >= 8:
            return "Great head position. Well balanced over front knee."
        elif score >= 6:
            return "Good head position. Try to keep head more still during swing."
        else:
            return "Focus on keeping head steady and over front knee throughout shot."
    
    def _get_swing_control_feedback(self, score: float, elbow_angles: List[float]) -> str:
        """Generate swing control feedback"""
        if score >= 8:
            return "Excellent bat swing control. Good elbow position maintained."
        elif score >= 6:
            return "Good swing mechanics. Work on maintaining consistent elbow height."
        else:
            return "Focus on keeping front elbow up and driving through the ball."
    
    def _get_balance_feedback(self, score: float, spine_leans: List[float]) -> str:
        """Generate balance feedback"""
        if score >= 8:
            return "Excellent balance maintained throughout the shot."
        elif score >= 6:
            return "Good balance. Try to stay more upright during execution."
        else:
            return "Work on maintaining better balance. Avoid leaning too much."
    
    def _get_follow_through_feedback(self, score: float) -> str:
        """Generate follow-through feedback"""
        if score >= 8:
            return "Strong follow-through. Good extension after contact."
        elif score >= 6:
            return "Decent follow-through. Ensure full extension toward target."
        else:
            return "Work on completing the shot with full follow-through motion."
    
    def _default_evaluation(self) -> ShotEvaluation:
        """Return default evaluation when no data available"""
        return ShotEvaluation(
            footwork_score=5.0,
            head_position_score=5.0,
            swing_control_score=5.0,
            balance_score=5.0,
            follow_through_score=5.0,
            overall_score=5.0,
            footwork_feedback="Unable to analyze footwork - insufficient pose data.",
            head_position_feedback="Unable to analyze head position - insufficient pose data.",
            swing_control_feedback="Unable to analyze swing control - insufficient pose data.",
            balance_feedback="Unable to analyze balance - insufficient pose data.",
            follow_through_feedback="Unable to analyze follow-through - insufficient pose data."
        )

class CricketAnalyzer:
    """Main cricket analysis system"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.pose_estimator = PoseEstimator()
        self.biomechanics_analyzer = BiomechanicsAnalyzer()
        self.overlay_renderer = VideoOverlayRenderer()
        self.shot_evaluator = ShotEvaluator()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.processing_times = []
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Main video analysis pipeline"""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Starting analysis of: {video_path}")
        self.start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        
        # Setup output video writer
        output_path = os.path.join(self.output_dir, "annotated_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start_time = time.time()
                
                # Calculate timestamp
                timestamp = frame_number / fps
                
                # Pose estimation
                pose_results, pose_confidence = self.pose_estimator.extract_keypoints(frame)
                
                # Biomechanical analysis
                if pose_results and pose_results.pose_landmarks:
                    metrics = self.biomechanics_analyzer.analyze_frame(
                        pose_results.pose_landmarks, 
                        frame.shape, 
                        frame_number, 
                        timestamp
                    )
                    metrics.pose_confidence = pose_confidence
                else:
                    # Create empty metrics for missing poses
                    metrics = BiomechanicalMetrics(frame_number=frame_number, timestamp=timestamp)
                
                # Add to evaluation history
                self.shot_evaluator.add_metrics(metrics)
                
                # Draw overlays
                annotated_frame = frame.copy()
                
                # Draw skeleton
                if pose_results and pose_results.pose_landmarks:
                    annotated_frame = self.overlay_renderer.draw_skeleton(
                        annotated_frame, pose_results.pose_landmarks, self.pose_estimator
                    )
                
                # Draw metrics overlay
                annotated_frame = self.overlay_renderer.draw_metrics_overlay(annotated_frame, metrics)
                
                # Write frame
                out.write(annotated_frame)
                
                # Performance tracking
                frame_time = time.time() - frame_start_time
                self.processing_times.append(frame_time)
                
                frame_number += 1
                
                # Progress logging
                if frame_number % 30 == 0:
                    avg_fps = 1.0 / np.mean(self.processing_times[-30:]) if self.processing_times else 0
                    progress = (frame_number / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% | Processing FPS: {avg_fps:.1f}")
                
        finally:
            cap.release()
            out.release()
        
        # Calculate final performance metrics
        total_time = time.time() - self.start_time
        avg_fps = frame_number / total_time if total_time > 0 else 0
        
        logger.info(f"Analysis complete: {frame_number} frames processed in {total_time:.1f}s")
        logger.info(f"Average processing FPS: {avg_fps:.1f}")
        
        # Generate final evaluation
        evaluation = self.shot_evaluator.evaluate_shot()
        
        # Save evaluation
        evaluation_path = os.path.join(self.output_dir, "evaluation.json")
        with open(evaluation_path, 'w') as f:
            json.dump(asdict(evaluation), f, indent=2)
        
        logger.info(f"Evaluation saved to: {evaluation_path}")
        logger.info(f"Annotated video saved to: {output_path}")
        
        return {
            "video_path": output_path,
            "evaluation_path": evaluation_path,
            "evaluation": asdict(evaluation),
            "performance": {
                "total_frames": frame_number,
                "processing_time": total_time,
                "avg_fps": avg_fps,
                "target_fps_achieved": avg_fps >= 10
            }
        }
