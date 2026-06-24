import { useEffect, useState } from 'react'
import type { RenderScale, SlicePlane, ViewerSettings } from './Viewer'
import { ChevronRightIcon } from './icons'

interface Props {
  settings: ViewerSettings
  onChange: (patch: Partial<ViewerSettings>) => void
  onClose: () => void
}

/** A small two-option segmented toggle. */
function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
}) {
  return (
    <div className="vs-segmented">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className={o.value === value ? 'vs-seg active' : 'vs-seg'}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

/** Viewer display settings popover (opened from the gear in the panel header). */
export function ViewerSettingsPanel({ settings, onChange, onClose }: Props) {
  // Advanced (rendering-fidelity) settings are set-once and technical, so they
  // start collapsed; the everyday display controls are always visible.
  const [advanced, setAdvanced] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <>
      <div className="vs-backdrop" onClick={onClose} />
      <div className="vs-popover" role="dialog" aria-label="Viewer settings">
        <div className="vs-row">
          <span className="vs-label" title="Image-left = patient-left (neuro) or patient-right (rad)">
            Orientation
          </span>
          <Segmented
            value={settings.radiological ? 'rad' : 'neuro'}
            options={[
              { value: 'neuro', label: 'Neuro' },
              { value: 'rad', label: 'Radiological' },
            ]}
            onChange={(v) => onChange({ radiological: v === 'rad' })}
          />
        </div>

        <div className="vs-row">
          <span className="vs-label">Layout</span>
          <select
            className="vs-select"
            value={settings.slicePlane}
            onChange={(e) => onChange({ slicePlane: e.target.value as SlicePlane })}
          >
            <option value="multiplanar">Multiplanar</option>
            <option value="axial">Axial</option>
            <option value="coronal">Coronal</option>
            <option value="sagittal">Sagittal</option>
          </select>
        </div>

        <label className="vs-row vs-check">
          <span className="vs-label" title="Show the 3D render alongside the slices (multiplanar only)">
            3D render
          </span>
          <input
            type="checkbox"
            checked={settings.showRender}
            disabled={settings.slicePlane !== 'multiplanar'}
            onChange={(e) => onChange({ showRender: e.target.checked })}
          />
        </label>

        <label className="vs-row vs-check">
          <span className="vs-label">Crosshair</span>
          <input
            type="checkbox"
            checked={settings.crosshair}
            onChange={(e) => onChange({ crosshair: e.target.checked })}
          />
        </label>

        <button
          type="button"
          className="vs-advanced-toggle"
          onClick={() => setAdvanced((v) => !v)}
        >
          <ChevronRightIcon size={12} className={advanced ? 'vs-chevron open' : 'vs-chevron'} />
          Advanced
        </button>

        {advanced && (
          <>
            <div className="vs-row">
              <span className="vs-label" title="Nearest = exact voxels & crisp labels; linear = smoothed">
                Interpolation
              </span>
              <Segmented
                value={settings.interpolation}
                options={[
                  { value: 'nearest', label: 'Nearest' },
                  { value: 'linear', label: 'Linear' },
                ]}
                onChange={(v) => onChange({ interpolation: v })}
              />
            </div>

            <div className="vs-row">
              <span className="vs-label" title="MSAA edge smoothing; uses more GPU memory">
                Anti-aliasing
              </span>
              <Segmented
                value={settings.antialias ? 'on' : 'off'}
                options={[
                  { value: 'off', label: 'Off' },
                  { value: 'on', label: 'On' },
                ]}
                onChange={(v) => onChange({ antialias: v === 'on' })}
              />
            </div>

            <div className="vs-row">
              <span className="vs-label" title="Supersample N× then downsample; Native matches your display">
                Render scale
              </span>
              <select
                className="vs-select"
                value={settings.renderScale}
                onChange={(e) => onChange({ renderScale: e.target.value as RenderScale })}
              >
                <option value="native">Native</option>
                <option value="2x">2×</option>
                <option value="4x">4×</option>
              </select>
            </div>
          </>
        )}
      </div>
    </>
  )
}
