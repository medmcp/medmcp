// The mark shown next to each model in the Models window: the model family's
// own where it has one, otherwise its publisher's. Bundled from an icon package
// rather than fetched, because the workspace has to work offline.
//
// These are trademarks of their owners, shown only to say whose model a row is.
import google from '@lobehub/icons-static-svg/icons/google-color.svg'
import ibm from '@lobehub/icons-static-svg/icons/ibm.svg'
import meta from '@lobehub/icons-static-svg/icons/meta-color.svg'
import mistral from '@lobehub/icons-static-svg/icons/mistral-color.svg'
import nvidia from '@lobehub/icons-static-svg/icons/nvidia-color.svg'
import qwen from '@lobehub/icons-static-svg/icons/qwen-color.svg'

export interface ModelLogo {
  src: string
  /** Single-colour artwork with no colours of its own; the UI tints it. */
  mono?: boolean
}

/** Catalog model id → its mark. A model without an entry simply has none. */
export const MODEL_LOGOS: Record<string, ModelLogo> = {
  'muse-glimmer': { src: meta },
  'devstral-small-2': { src: mistral },
  // Gemma has a mark of its own, but it is thin dark-blue linework that all but
  // disappears on this surface; Google's is legible at this size.
  gemma4: { src: google },
  // IBM's logo is one colour by design, so there is no colour file for it.
  'granite4.2': { src: ibm, mono: true },
  'mistral-small3.2': { src: mistral },
  'nemotron-3.5-lightning': { src: nvidia },
  'qwen3.8': { src: qwen },
}
