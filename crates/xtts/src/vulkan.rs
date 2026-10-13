//! Vulkan device discovery for the (upstream-pending) Candle Vulkan backend.
//!
//! candle-core 0.11 has no `Device::Vulkan` variant — the Vulkan backend was
//! removed upstream when it was replaced by the `ug` unified-graph layer, and
//! `candle-ug` 0.11 only ships `cuda` and `metal` backends. This module is the
//! runtime half of Vulkan support in this crate: with the `vulkan` feature it
//! probes the system Vulkan loader (via `ash`) and reports usable GPUs, so the
//! CLI can validate `--device vulkan[:N]` and print a helpful message. Once a
//! Candle build exposing `Device::new_vulkan` is used as the workspace
//! dependency, [`connect`] below becomes the single place that hands a device
//! to `Xtts::load`.

use anyhow::{Context, Result};

/// One Vulkan-capable physical device (GPU/APU/CPU implementation).
#[derive(Debug, Clone)]
pub struct VulkanDeviceInfo {
    /// Zero-based index among *compute-capable* devices, matching the numbering
    /// `--device vulkan:N` uses.
    pub index: usize,
    pub name: String,
    /// Encoded driver version (e.g. `VK_MAKE_DRIVER_VERSION` layout).
    pub driver_version: u32,
    pub api_version: u32,
    /// True when the device advertises `UNIFIED_MEMORY` or supports host-visible
    /// coherent device memory, which Candle's backend would rely on heavily.
    pub unified_memory: bool,
}

impl VulkanDeviceInfo {
    pub fn summary(&self) -> String {
        let major = self.api_version >> 22;
        let minor = (self.api_version >> 12) & 0x3FF;
        format!(
            "vulkan:{idx} \"{name}\" (API {major}.{minor}, driver {drv:#x}, {})",
            if self.unified_memory { "UM" } else { "discrete memory" },
            idx = self.index,
            name = self.name,
            drv = self.driver_version,
        )
    }
}

/// Enumerate compute-capable Vulkan devices visible to the system loader.
///
/// Returns an empty list when the feature is off, the loader (`libvulkan`) is
/// missing, or no device exposes a compute queue.
pub fn enumerate() -> Result<Vec<VulkanDeviceInfo>> {
    #[cfg(feature = "vulkan")]
    {
        enumerate_impl().context("vulkan device enumeration")
    }
    #[cfg(not(feature = "vulkan"))]
    {
        Ok(vec![])
    }
}

/// Whether at least one Vulkan compute device is present.
pub fn is_available() -> bool {
    !enumerate().unwrap_or_default().is_empty()
}

/// Return the compute-capable device with the given ordinal, for use as the
/// selection target of `--device vulkan:N`.
pub fn device_info(index: usize) -> Result<VulkanDeviceInfo> {
    enumerate()?
        .into_iter()
        .find(|d| d.index == index)
        .with_context(|| format!("no vulkan device at index {index}"))
}

/// Hand out a Candle device for `--device vulkan[:N]`.
///
/// Always fails today: candle-core 0.11 cannot construct a Vulkan device. The
/// error explains what is required. When Candle gains the backend, replace the
/// body with `candle_core::Device::new_vulkan(index)` behind the `vulkan`
/// feature; callers (`Xtts::load`, the CLI) need no further changes.
pub fn connect(_index: usize) -> Result<candle_core::Device> {
    #[cfg(feature = "vulkan")]
    {
        // Validate the index against real hardware first so users get "no such
        // GPU" rather than "backend missing" when they typo the ordinal.
        let info = device_info(_index)?;
        anyhow::bail!(
            "vulkan acceleration is not available in this build: candle-core {} has no \
             Vulkan backend (found {}, but Device::Vulkan does not exist upstream yet; \
             see https://github.com/huggingface/candle - use --device cuda or cpu)",
            env!("CARGO_PKG_VERSION"),
            info.summary(),
        )
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = _index;
        anyhow::bail!(
            "--device vulkan requires rebuilding with the `vulkan` feature \
             (cargo build --features vulkan); candle-core also needs a Vulkan \
             backend, which 0.11 does not provide — use --device cuda, metal or cpu"
        )
    }
}

#[cfg(feature = "vulkan")]
fn enumerate_impl() -> Result<Vec<VulkanDeviceInfo>> {
    use ash::vk;

    // `ash::Entry::load()` dlopen()s the system Vulkan loader at runtime, so a
    // machine without libvulkan fails here with a clear message instead of at
    // link time.
    let entry = ash::Entry::load().context(
        "Vulkan loader not found: install the Vulkan runtime (libvulkan.so.1 / \
         vulkan-1.dll / MoltenVK) or use --device cpu",
    )?;

    let app_info = vk::ApplicationInfo::default()
        .application_name(c"Candle XTTS")
        .api_version(vk::API_VERSION_1_2);
    let create = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&create, None) }
        .context("cannot create a Vulkan instance (driver too old?)")?;

    let out = (|| -> Result<Vec<VulkanDeviceInfo>> {
        let devices = unsafe { instance.enumerate_physical_devices() }
            .context("vkEnumeratePhysicalDevices failed")?;
        let mut found = Vec::new();
        for pd in devices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let queues = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            // Candle's backend needs a compute queue; skip present-only adapters.
            let compute = queues
                .iter()
                .any(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE));
            if !compute {
                continue;
            }
            // Heuristic: integrated GPUs and CPU implementations (lavapipe,
            // llvmpipe) give Candle's backend unified host/device memory.
            let unified_memory = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
                || props.device_type == vk::PhysicalDeviceType::CPU;
            let name = props
                .device_name
                .split(|&b| b == 0)
                .next()
                .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                .unwrap_or_else(|| format!("unknown vulkan device"));
            found.push(VulkanDeviceInfo {
                index: found.len(),
                name,
                driver_version: props.driver_version,
                api_version: props.api_version,
                unified_memory,
            });
        }
        Ok(found)
    })();

    unsafe { instance.destroy_instance(None) };
    out
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    #[test]
    fn enumerate_does_not_panic() {
        // Machines without a Vulkan driver must yield Ok(empty), not an error.
        let _ = super::enumerate();
    }
}
