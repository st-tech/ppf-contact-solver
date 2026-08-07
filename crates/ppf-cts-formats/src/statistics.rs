// File: crates/ppf-cts-formats/src/statistics.rs
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

//! Versioned per-object timeline statistics produced by the solver.
//!
//! A manifest fixes object identity, ordering, and supported scalar channels
//! for one run. Each output frame then carries one record in that exact order.
//! The validity mask is separate from the values so an unavailable metric is
//! never confused with a physical zero or represented by a control-flow NaN.

use serde::{Deserialize, Serialize};

use crate::envelope::{from_cbor_with_version, to_cbor_with_version, FormatError};

pub const STATISTICS_VERSION: u32 = 1;
pub const KIND_STATISTICS_MANIFEST: &str = "StatisticsManifest";
pub const KIND_STATISTICS_FRAME: &str = "StatisticsFrame";
pub const KIND_STATISTICS_INPUT: &str = "StatisticsInput";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StatisticChannel {
    LocationX = 0,
    LocationY = 1,
    LocationZ = 2,
    Volume = 3,
    SurfaceArea = 4,
    AreaStretch = 5,
    RodLength = 6,
    LengthStretch = 7,
    VelocityX = 8,
    VelocityY = 9,
    VelocityZ = 10,
    Speed = 11,
    AccelerationX = 12,
    AccelerationY = 13,
    AccelerationZ = 14,
    AccelerationMagnitude = 15,
    AngularVelocityX = 16,
    AngularVelocityY = 17,
    AngularVelocityZ = 18,
    AngularSpeed = 19,
    AngularAxisX = 20,
    AngularAxisY = 21,
    AngularAxisZ = 22,
    VolumeStretch = 23,
    ContactCount = 24,
}

impl StatisticChannel {
    pub const ALL: [Self; 25] = [
        Self::LocationX,
        Self::LocationY,
        Self::LocationZ,
        Self::Volume,
        Self::SurfaceArea,
        Self::AreaStretch,
        Self::RodLength,
        Self::LengthStretch,
        Self::VelocityX,
        Self::VelocityY,
        Self::VelocityZ,
        Self::Speed,
        Self::AccelerationX,
        Self::AccelerationY,
        Self::AccelerationZ,
        Self::AccelerationMagnitude,
        Self::AngularVelocityX,
        Self::AngularVelocityY,
        Self::AngularVelocityZ,
        Self::AngularSpeed,
        Self::AngularAxisX,
        Self::AngularAxisY,
        Self::AngularAxisZ,
        Self::VolumeStretch,
        Self::ContactCount,
    ];

    pub const fn bit(self) -> u64 {
        1_u64 << self as u8
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::LocationX => "location_x",
            Self::LocationY => "location_y",
            Self::LocationZ => "location_z",
            Self::Volume => "volume",
            Self::SurfaceArea => "surface_area",
            Self::AreaStretch => "area_stretch",
            Self::RodLength => "rod_length",
            Self::LengthStretch => "length_stretch",
            Self::VelocityX => "velocity_x",
            Self::VelocityY => "velocity_y",
            Self::VelocityZ => "velocity_z",
            Self::Speed => "speed",
            Self::AccelerationX => "acceleration_x",
            Self::AccelerationY => "acceleration_y",
            Self::AccelerationZ => "acceleration_z",
            Self::AccelerationMagnitude => "acceleration_magnitude",
            Self::AngularVelocityX => "angular_velocity_x",
            Self::AngularVelocityY => "angular_velocity_y",
            Self::AngularVelocityZ => "angular_velocity_z",
            Self::AngularSpeed => "angular_speed",
            Self::AngularAxisX => "angular_axis_x",
            Self::AngularAxisY => "angular_axis_y",
            Self::AngularAxisZ => "angular_axis_z",
            Self::VolumeStretch => "volume_stretch",
            Self::ContactCount => "contact_count",
        }
    }
}

pub const ALL_STATISTIC_CHANNELS: u64 = (1_u64 << StatisticChannel::ALL.len()) - 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatisticsManifest {
    pub objects: Vec<StatisticsObject>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatisticsObject {
    pub object_index: u32,
    pub object_uuid: String,
    pub object_name: String,
    pub dynamics_type: String,
    pub supported_channels: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatisticsInput {
    pub objects: Vec<StatisticsInputObject>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatisticsInputObject {
    pub object_index: u32,
    pub object_uuid: String,
    pub object_name: String,
    pub dynamics_type: String,
    #[serde(default)]
    pub static_object: bool,
    pub vertex_indices: Vec<u32>,
    /// `[start, count]` in the solver's global rod array.
    pub rod_range: [u32; 2],
    /// `[start, count]` in the solver's global face array.
    pub face_range: [u32; 2],
    /// `[start, count]` in the solver's global tet array.
    pub tet_range: [u32; 2],
    pub closed_surface: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grain_radius: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatisticsFrame {
    pub solver_frame: u32,
    pub time_seconds: f64,
    pub objects: Vec<ObjectStatistics>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectStatistics {
    pub object_index: u32,
    pub location: [f32; 3],
    pub volume: f32,
    pub volume_stretch: f32,
    pub surface_area: f32,
    pub area_stretch: f32,
    pub rod_length: f32,
    pub length_stretch: f32,
    pub velocity: [f32; 3],
    pub speed: f32,
    pub acceleration: [f32; 3],
    pub acceleration_magnitude: f32,
    pub angular_velocity: [f32; 3],
    pub angular_speed: f32,
    pub angular_axis: [f32; 3],
    pub contact_count: u64,
    pub valid_channels: u64,
}

impl Default for ObjectStatistics {
    fn default() -> Self {
        Self {
            object_index: 0,
            location: [0.0; 3],
            volume: 0.0,
            volume_stretch: 0.0,
            surface_area: 0.0,
            area_stretch: 0.0,
            rod_length: 0.0,
            length_stretch: 0.0,
            velocity: [0.0; 3],
            speed: 0.0,
            acceleration: [0.0; 3],
            acceleration_magnitude: 0.0,
            angular_velocity: [0.0; 3],
            angular_speed: 0.0,
            angular_axis: [0.0; 3],
            contact_count: 0,
            valid_channels: 0,
        }
    }
}

impl ObjectStatistics {
    pub fn scalar(&self, channel: StatisticChannel) -> Option<f64> {
        if self.valid_channels & channel.bit() == 0 {
            return None;
        }
        let value = match channel {
            StatisticChannel::LocationX => self.location[0] as f64,
            StatisticChannel::LocationY => self.location[1] as f64,
            StatisticChannel::LocationZ => self.location[2] as f64,
            StatisticChannel::Volume => self.volume as f64,
            StatisticChannel::VolumeStretch => self.volume_stretch as f64,
            StatisticChannel::SurfaceArea => self.surface_area as f64,
            StatisticChannel::AreaStretch => self.area_stretch as f64,
            StatisticChannel::RodLength => self.rod_length as f64,
            StatisticChannel::LengthStretch => self.length_stretch as f64,
            StatisticChannel::VelocityX => self.velocity[0] as f64,
            StatisticChannel::VelocityY => self.velocity[1] as f64,
            StatisticChannel::VelocityZ => self.velocity[2] as f64,
            StatisticChannel::Speed => self.speed as f64,
            StatisticChannel::AccelerationX => self.acceleration[0] as f64,
            StatisticChannel::AccelerationY => self.acceleration[1] as f64,
            StatisticChannel::AccelerationZ => self.acceleration[2] as f64,
            StatisticChannel::AccelerationMagnitude => self.acceleration_magnitude as f64,
            StatisticChannel::AngularVelocityX => self.angular_velocity[0] as f64,
            StatisticChannel::AngularVelocityY => self.angular_velocity[1] as f64,
            StatisticChannel::AngularVelocityZ => self.angular_velocity[2] as f64,
            StatisticChannel::AngularSpeed => self.angular_speed as f64,
            StatisticChannel::AngularAxisX => self.angular_axis[0] as f64,
            StatisticChannel::AngularAxisY => self.angular_axis[1] as f64,
            StatisticChannel::AngularAxisZ => self.angular_axis[2] as f64,
            StatisticChannel::ContactCount => self.contact_count as f64,
        };
        Some(value)
    }

    fn finite_float_channels(&self) -> [(StatisticChannel, f32); 24] {
        [
            (StatisticChannel::LocationX, self.location[0]),
            (StatisticChannel::LocationY, self.location[1]),
            (StatisticChannel::LocationZ, self.location[2]),
            (StatisticChannel::Volume, self.volume),
            (StatisticChannel::SurfaceArea, self.surface_area),
            (StatisticChannel::AreaStretch, self.area_stretch),
            (StatisticChannel::RodLength, self.rod_length),
            (StatisticChannel::LengthStretch, self.length_stretch),
            (StatisticChannel::VelocityX, self.velocity[0]),
            (StatisticChannel::VelocityY, self.velocity[1]),
            (StatisticChannel::VelocityZ, self.velocity[2]),
            (StatisticChannel::Speed, self.speed),
            (StatisticChannel::AccelerationX, self.acceleration[0]),
            (StatisticChannel::AccelerationY, self.acceleration[1]),
            (StatisticChannel::AccelerationZ, self.acceleration[2]),
            (
                StatisticChannel::AccelerationMagnitude,
                self.acceleration_magnitude,
            ),
            (StatisticChannel::AngularVelocityX, self.angular_velocity[0]),
            (StatisticChannel::AngularVelocityY, self.angular_velocity[1]),
            (StatisticChannel::AngularVelocityZ, self.angular_velocity[2]),
            (StatisticChannel::AngularSpeed, self.angular_speed),
            (StatisticChannel::AngularAxisX, self.angular_axis[0]),
            (StatisticChannel::AngularAxisY, self.angular_axis[1]),
            (StatisticChannel::AngularAxisZ, self.angular_axis[2]),
            (StatisticChannel::VolumeStretch, self.volume_stretch),
        ]
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StatisticsValidationError {
    #[error("statistics manifest object {position} has index {found}, expected {expected}")]
    ManifestObjectOrder {
        position: usize,
        found: u32,
        expected: u32,
    },
    #[error("statistics manifest object {object_index} has an empty UUID")]
    EmptyObjectUuid { object_index: u32 },
    #[error("statistics manifest object {object_index} enables unknown channel bits {bits:#x}")]
    UnknownSupportedChannels { object_index: u32, bits: u64 },
    #[error("statistics frame object count is {found}, expected {expected}")]
    ObjectCount { found: usize, expected: usize },
    #[error("statistics frame object {position} has index {found}, expected {expected}")]
    FrameObjectOrder {
        position: usize,
        found: u32,
        expected: u32,
    },
    #[error("statistics frame object {object_index} marks unsupported channels valid: {bits:#x}")]
    UnsupportedValidChannels { object_index: u32, bits: u64 },
    #[error("statistics frame object {object_index} enables unknown validity bits {bits:#x}")]
    UnknownValidChannels { object_index: u32, bits: u64 },
    #[error("statistics frame object {object_index} has non-finite valid channel {channel}")]
    NonFiniteValue {
        object_index: u32,
        channel: &'static str,
    },
    #[error("statistics frame time is not finite or is negative")]
    InvalidTime,
}

impl StatisticsManifest {
    pub fn validate(&self) -> Result<(), StatisticsValidationError> {
        for (position, object) in self.objects.iter().enumerate() {
            let expected = position as u32;
            if object.object_index != expected {
                return Err(StatisticsValidationError::ManifestObjectOrder {
                    position,
                    found: object.object_index,
                    expected,
                });
            }
            if object.object_uuid.is_empty() {
                return Err(StatisticsValidationError::EmptyObjectUuid {
                    object_index: object.object_index,
                });
            }
            let unknown = object.supported_channels & !ALL_STATISTIC_CHANNELS;
            if unknown != 0 {
                return Err(StatisticsValidationError::UnknownSupportedChannels {
                    object_index: object.object_index,
                    bits: unknown,
                });
            }
        }
        Ok(())
    }
}

impl StatisticsFrame {
    pub fn validate(&self, manifest: &StatisticsManifest) -> Result<(), StatisticsValidationError> {
        manifest.validate()?;
        if !self.time_seconds.is_finite() || self.time_seconds < 0.0 {
            return Err(StatisticsValidationError::InvalidTime);
        }
        if self.objects.len() != manifest.objects.len() {
            return Err(StatisticsValidationError::ObjectCount {
                found: self.objects.len(),
                expected: manifest.objects.len(),
            });
        }
        for (position, (record, object)) in self.objects.iter().zip(&manifest.objects).enumerate() {
            if record.object_index != object.object_index {
                return Err(StatisticsValidationError::FrameObjectOrder {
                    position,
                    found: record.object_index,
                    expected: object.object_index,
                });
            }
            let unknown = record.valid_channels & !ALL_STATISTIC_CHANNELS;
            if unknown != 0 {
                return Err(StatisticsValidationError::UnknownValidChannels {
                    object_index: record.object_index,
                    bits: unknown,
                });
            }
            let unsupported = record.valid_channels & !object.supported_channels;
            if unsupported != 0 {
                return Err(StatisticsValidationError::UnsupportedValidChannels {
                    object_index: record.object_index,
                    bits: unsupported,
                });
            }
            for (channel, value) in record.finite_float_channels() {
                if record.valid_channels & channel.bit() != 0 && !value.is_finite() {
                    return Err(StatisticsValidationError::NonFiniteValue {
                        object_index: record.object_index,
                        channel: channel.name(),
                    });
                }
            }
        }
        Ok(())
    }
}

pub fn encode_statistics_manifest(manifest: &StatisticsManifest) -> Result<Vec<u8>, FormatError> {
    to_cbor_with_version(STATISTICS_VERSION, KIND_STATISTICS_MANIFEST, manifest)
}

pub fn decode_statistics_manifest(bytes: &[u8]) -> Result<StatisticsManifest, FormatError> {
    from_cbor_with_version(STATISTICS_VERSION, KIND_STATISTICS_MANIFEST, bytes)
}

pub fn encode_statistics_frame(frame: &StatisticsFrame) -> Result<Vec<u8>, FormatError> {
    to_cbor_with_version(STATISTICS_VERSION, KIND_STATISTICS_FRAME, frame)
}

pub fn decode_statistics_frame(bytes: &[u8]) -> Result<StatisticsFrame, FormatError> {
    from_cbor_with_version(STATISTICS_VERSION, KIND_STATISTICS_FRAME, bytes)
}

pub fn encode_statistics_input(input: &StatisticsInput) -> Result<Vec<u8>, FormatError> {
    to_cbor_with_version(STATISTICS_VERSION, KIND_STATISTICS_INPUT, input)
}

pub fn decode_statistics_input(bytes: &[u8]) -> Result<StatisticsInput, FormatError> {
    from_cbor_with_version(STATISTICS_VERSION, KIND_STATISTICS_INPUT, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> StatisticsManifest {
        StatisticsManifest {
            objects: vec![StatisticsObject {
                object_index: 0,
                object_uuid: "object-uuid".into(),
                object_name: "Cloth".into(),
                dynamics_type: "SHELL".into(),
                supported_channels: StatisticChannel::LocationX.bit()
                    | StatisticChannel::AreaStretch.bit()
                    | StatisticChannel::ContactCount.bit(),
            }],
        }
    }

    fn frame() -> StatisticsFrame {
        StatisticsFrame {
            solver_frame: 2,
            time_seconds: 1.0 / 30.0,
            objects: vec![ObjectStatistics {
                object_index: 0,
                location: [1.5, 0.0, 0.0],
                area_stretch: 1.02,
                contact_count: 7,
                valid_channels: StatisticChannel::LocationX.bit()
                    | StatisticChannel::AreaStretch.bit()
                    | StatisticChannel::ContactCount.bit(),
                ..Default::default()
            }],
        }
    }

    #[test]
    fn manifest_and_frame_roundtrip() {
        let manifest = manifest();
        let frame = frame();
        manifest.validate().unwrap();
        frame.validate(&manifest).unwrap();

        let manifest_bytes = encode_statistics_manifest(&manifest).unwrap();
        let frame_bytes = encode_statistics_frame(&frame).unwrap();
        assert_eq!(
            decode_statistics_manifest(&manifest_bytes).unwrap(),
            manifest
        );
        assert_eq!(decode_statistics_frame(&frame_bytes).unwrap(), frame);
    }

    #[test]
    fn scalar_honors_validity() {
        let record = &frame().objects[0];
        assert_eq!(
            record.scalar(StatisticChannel::AreaStretch),
            Some(1.02_f32 as f64)
        );
        assert_eq!(record.scalar(StatisticChannel::Volume), None);
        assert_eq!(record.scalar(StatisticChannel::ContactCount), Some(7.0));
    }

    #[test]
    fn rejects_wrong_record_order() {
        let mut frame = frame();
        frame.objects[0].object_index = 4;
        assert!(matches!(
            frame.validate(&manifest()),
            Err(StatisticsValidationError::FrameObjectOrder { .. })
        ));
    }

    #[test]
    fn rejects_unsupported_valid_channel() {
        let mut frame = frame();
        frame.objects[0].valid_channels |= StatisticChannel::Volume.bit();
        assert!(matches!(
            frame.validate(&manifest()),
            Err(StatisticsValidationError::UnsupportedValidChannels { .. })
        ));
    }

    #[test]
    fn rejects_non_finite_valid_value() {
        let mut frame = frame();
        frame.objects[0].area_stretch = f32::NAN;
        assert!(matches!(
            frame.validate(&manifest()),
            Err(StatisticsValidationError::NonFiniteValue {
                channel: "area_stretch",
                ..
            })
        ));
    }

    #[test]
    fn input_roundtrip() {
        let input = StatisticsInput {
            objects: vec![StatisticsInputObject {
                object_index: 0,
                object_uuid: "object-uuid".into(),
                object_name: "Cloth".into(),
                dynamics_type: "SHELL".into(),
                static_object: false,
                vertex_indices: vec![0, 1, 2],
                rod_range: [0, 0],
                face_range: [0, 1],
                tet_range: [0, 0],
                closed_surface: false,
                grain_radius: None,
            }],
        };
        let bytes = encode_statistics_input(&input).unwrap();
        assert_eq!(decode_statistics_input(&bytes).unwrap(), input);
    }
}
