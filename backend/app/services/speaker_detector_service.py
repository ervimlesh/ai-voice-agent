import librosa
import numpy as np
import soundfile as sf
from typing import List, Tuple, Optional
from app.models.speaker_profile import SpeakerProfile

  
class SpeakerDetector:
    """Service for extracting voice features and detecting speakers"""
    
    def __init__(self):
        self.sample_rate = 16000  # Standard sample rate for voice processing
        self.mfcc_coefficients = 13  # Number of MFCC coefficients to extract
    
    def extract_features(self, audio_path: str) -> SpeakerProfile:
        """Extract voice features from audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract fundamental frequency (pitch)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            pitch = float(np.mean(pitch_values)) if pitch_values else 0.0
            
            # Extract spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid = float(np.mean(spectral_centroids))
            
            # Extract MFCC coefficients
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.mfcc_coefficients)
            mfcc_mean = np.mean(mfccs, axis=1).tolist()
            
            # Extract zero crossing rate (voice quality)
            zero_crossing_rates = librosa.feature.zero_crossing_rate(y)[0]
            zero_crossing_rate = float(np.mean(zero_crossing_rates))
            
            # Extract RMS energy (loudness)
            rms_energies = librosa.feature.rms(y=y)[0]
            rms_energy = float(np.mean(rms_energies))
            
            # Calculate speech rate (words per minute) - simplified version
            # This is a basic approximation using spectral flux
            speech_rate = self._estimate_speech_rate(y, sr)
            
            return SpeakerProfile(
                pitch=pitch,
                spectral_centroid=spectral_centroid,
                mfcc=mfcc_mean,
                zero_crossing_rate=zero_crossing_rate,
                rms_energy=rms_energy,
                speech_rate=speech_rate
            )
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            # Return default profile on error
            return SpeakerProfile(
                pitch=0.0,
                spectral_centroid=0.0,
                mfcc=[0.0] * self.mfcc_coefficients,
                zero_crossing_rate=0.0,
                rms_energy=0.0,
                speech_rate=0.0
            )
    
    def extract_features_from_array(self, y: np.ndarray, sr: int = 16000) -> SpeakerProfile:
        """Extract voice features directly from an in-memory float32 slice.

        Used for per-turn sub-segments so we don't write a temp WAV for each one.
        """
        try:
            if y is None or len(y) == 0:
                return self._empty_profile()
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
            # Too short to characterize reliably
            if len(y) < int(0.2 * sr):
                return self._empty_profile()

            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                p = pitches[index, t]
                if p > 0:
                    pitch_values.append(p)
            pitch = float(np.mean(pitch_values)) if pitch_values else 0.0

            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.mfcc_coefficients)
            mfcc_mean = np.mean(mfccs, axis=1).tolist()
            zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
            rms_energy = float(np.mean(librosa.feature.rms(y=y)[0]))
            speech_rate = self._estimate_speech_rate(y, sr)
            embedding = self.build_embedding(y, sr)

            return SpeakerProfile(
                pitch=pitch,
                spectral_centroid=spectral_centroid,
                mfcc=mfcc_mean,
                zero_crossing_rate=zero_crossing_rate,
                rms_energy=rms_energy,
                speech_rate=speech_rate,
                embedding=embedding,
            )
        except Exception as e:
            print(f"Error extracting features from array: {str(e)}")
            return self._empty_profile()

    def build_embedding(self, y: np.ndarray, sr: int = 16000) -> List[float]:
        """Build an L2-normalized speaker embedding from raw audio.

        Tier B identity vector: concatenated mean/std of MFCC + delta MFCC.
        Captures voice timbre far better than scalar pitch comparison, and is
        cheap enough to run per sliding-window during lightweight diarization.
        """
        try:
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                sr = self.sample_rate
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            delta = librosa.feature.delta(mfcc)
            feat = np.concatenate([
                mfcc.mean(axis=1), mfcc.std(axis=1),
                delta.mean(axis=1), delta.std(axis=1),
            ])
            norm = np.linalg.norm(feat)
            return (feat / norm).tolist() if norm > 0 else feat.tolist()
        except Exception as e:
            print(f"Error building embedding: {str(e)}")
            return [0.0] * 80

    def _empty_profile(self) -> SpeakerProfile:
        return SpeakerProfile(
            pitch=0.0,
            spectral_centroid=0.0,
            mfcc=[0.0] * self.mfcc_coefficients,
            zero_crossing_rate=0.0,
            rms_energy=0.0,
            speech_rate=0.0,
            embedding=None,
        )

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity in range [-1, 1] between two embedding vectors."""
        if a is None or b is None:
            return -1.0
        va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
        if va.shape != vb.shape:
            return -1.0
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return -1.0
        return float(np.dot(va, vb) / (na * nb))

    def _estimate_speech_rate(self, y: np.ndarray, sr: int) -> float:
        """Estimate speech rate in words per minute (simplified)"""
        try:
            # Use onset detection to estimate speech rate
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) > 1:
                duration = len(y) / sr  # Duration in seconds
                onsets_per_second = len(onset_frames) / duration
                # Rough approximation: assume each onset could be a word start
                words_per_minute = (onsets_per_second * 60) / 2  # Divide by 2 as rough estimate
                return float(min(words_per_minute, 200))  # Cap at reasonable maximum
            return 0.0
        except:
            return 0.0
    
    def calculate_similarity(self, profile1: SpeakerProfile, profile2: SpeakerProfile) -> float:
        """Calculate similarity score between two speaker profiles (0-100)"""
        try:
            # Normalize features for comparison
            pitch_similarity = self._calculate_feature_similarity(profile1.pitch, profile2.pitch, 50, 400)
            spectral_similarity = self._calculate_feature_similarity(profile1.spectral_centroid, profile2.spectral_centroid, 1000, 5000)
            
            # Calculate MFCC similarity using cosine similarity
            mfcc_similarity = self._calculate_mfcc_similarity(profile1.mfcc, profile2.mfcc)
            
            # Calculate other feature similarities
            zcr_similarity = self._calculate_feature_similarity(profile1.zero_crossing_rate, profile2.zero_crossing_rate, 0, 0.5)
            rms_similarity = self._calculate_feature_similarity(profile1.rms_energy, profile2.rms_energy, 0, 1)
            
            # Weighted average of all similarities
            # MFCC and pitch are most important for speaker identification
            similarity_score = (
                mfcc_similarity * 0.4 +
                pitch_similarity * 0.3 +
                spectral_similarity * 0.15 +
                zcr_similarity * 0.1 +
                rms_similarity * 0.05
            )
            
            return float(similarity_score)
            
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def _calculate_feature_similarity(self, value1: float, value2: float, min_val: float, max_val: float) -> float:
        """Calculate similarity between two feature values (0-100)"""
        if min_val == max_val:
            return 100.0 if value1 == value2 else 0.0
        
        # Normalize values to 0-1 range
        norm1 = (value1 - min_val) / (max_val - min_val)
        norm2 = (value2 - min_val) / (max_val - min_val)
        
        # Calculate similarity as inverse of distance
        distance = abs(norm1 - norm2)
        similarity = max(0, 1 - distance) * 100
        
        return similarity
    
    def _calculate_mfcc_similarity(self, mfcc1: List[float], mfcc2: List[float]) -> float:
        """Calculate similarity between MFCC vectors using cosine similarity"""
        try:
            if len(mfcc1) != len(mfcc2):
                return 0.0
            
            # Convert to numpy arrays
            vec1 = np.array(mfcc1)
            vec2 = np.array(mfcc2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_similarity = dot_product / (norm1 * norm2)
            # Convert from [-1, 1] to [0, 100]
            similarity = (cosine_similarity + 1) * 50
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating MFCC similarity: {str(e)}")
            return 0.0
    
    def detect_speaker(self, audio_path: str, known_profiles: List[Tuple[str, SpeakerProfile]]) -> Tuple[Optional[str], float]:
        """
        Detect which known speaker is in the audio
        Returns: (speaker_id, confidence_score)
        """
        try:
            # Extract features from current audio
            current_profile = self.extract_features(audio_path)
            
            if not known_profiles:
                return None, 0.0
            
            # Compare with all known profiles
            best_match = None
            best_score = 0.0
            
            for speaker_id, profile in known_profiles:
                similarity = self.calculate_similarity(current_profile, profile)
                if similarity > best_score:
                    best_score = similarity
                    best_match = speaker_id
            
            return best_match, best_score
            
        except Exception as e:
            print(f"Error detecting speaker: {str(e)}")
            return None, 0.0
    
    def is_same_speaker(self, profile1: SpeakerProfile, profile2: SpeakerProfile, threshold: float = 95.0) -> bool:
        """Determine if two profiles represent the same speaker"""
        similarity = self.calculate_similarity(profile1, profile2)
        return similarity >= threshold
  