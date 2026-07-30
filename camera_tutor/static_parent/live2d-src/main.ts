/**
 * Camera Tutor — Live2D face control with viseme mapping.
 */
import { CubismFramework } from '@framework/live2dcubismframework';
import { LAppDelegate } from './lappdelegate';

let _mouthTarget = 0;
let _visemeType = 'sil';
let _initOk = false;
let _mouthId: any = null;
let _vowelIds: Record<string, any> = {};
let _stillIds: any[] = [];
let _frameCount = 0;
let _mouthCurrent = 0;

// Viseme → Kei vowel param mapping
const VISEME_VOWEL: Record<string, [string, number]> = {
  aa:  ['ParamA', 1.0],
  ae:  ['ParamA', 0.8],
  ah:  ['ParamA', 0.8],
  eh:  ['ParamE', 1.0],
  er:  ['ParamE', 0.5],
  iy:  ['ParamI', 1.0],
  ih:  ['ParamI', 0.7],
  uw:  ['ParamU', 1.0],
  ow:  ['ParamO', 1.0],
  ao:  ['ParamO', 0.8],
  aw:  ['ParamA', 0.6],
  oy:  ['ParamO', 0.5],
  ay:  ['ParamA', 0.7],
  h:   ['ParamA', 0.4],
  r:   ['ParamU', 0.3],
  l:   ['ParamI', 0.4],
  sz:  ['ParamI', 0.2],
  sh:  ['ParamU', 0.6],
  zh:  ['ParamU', 0.6],
  th:  ['ParamI', 0.3],
  dh:  ['ParamI', 0.3],
  fv:  ['ParamU', 0.2],
  td:  ['ParamI', 0.2],
  kg:  ['ParamE', 0.3],
  pb:  ['ParamU', 0.1],
};

(window as any)._setMouthOpen = (open: number, viseme?: string) => {
  _mouthTarget = open;
  if (viseme) _visemeType = viseme;
};

(function _applyFace() {
  _frameCount++;
  if (!_initOk) {
    const d: any = LAppDelegate.getInstance();
    const m: any = d._subdelegates?.[0]?._live2dManager?._models?.[0]?.getModel?.();
    if (!m) { requestAnimationFrame(_applyFace); return; }
    _mouthId = CubismFramework.getIdManager().getId('ParamMouthOpenY');
    for (const k of ['ParamA','ParamI','ParamU','ParamE','ParamO']) {
      _vowelIds[k] = CubismFramework.getIdManager().getId(k);
    }
    _stillIds = ['ParamAngleX','ParamAngleY','ParamAngleZ',
                 'ParamBodyAngleX','ParamBodyAngleY','ParamBodyAngleZ']
      .map(s => CubismFramework.getIdManager().getId(s));
    (window as any)._model = m;
    (window as any)._CF = CubismFramework; // for Console testing
    console.log('[Live2D] ready, vowels:', Object.keys(_vowelIds));
    _initOk = true;
  }
  const d: any = LAppDelegate.getInstance();
  const m: any = d._subdelegates?.[0]?._live2dManager?._models?.[0]?.getModel?.();
  if (m) {
    // Neutralize body sway
    for (const id of _stillIds) m.setParameterValueById(id, 0);
    // Mouth open amount
    const scaled = _mouthTarget < 0.06 ? 0 : Math.min(1, _mouthTarget * 2.5);
    m.setParameterValueById(_mouthId, scaled);
    // Vowel shape based on viseme
    const [vowelParam, strength] = VISEME_VOWEL[_visemeType] || ['', 0];
    // Reset all vowels to 0
    for (const id of Object.values(_vowelIds)) m.setParameterValueById(id as any, 0);
    // Set target vowel
    if (vowelParam && _vowelIds[vowelParam]) {
      m.setParameterValueById(_vowelIds[vowelParam], strength * Math.min(1, scaled * 2));
    }
    m.saveParameters();
    if (_frameCount % 120 === 0) {
      console.log('[Live2D]', _visemeType, 'open:', _mouthTarget.toFixed(2),
        'vowel:', vowelParam, '×', strength.toFixed(1));
    }
  }
  requestAnimationFrame(_applyFace);
})();

window.addEventListener('load', (): void => {
  if (!LAppDelegate.getInstance().initialize()) return;
  LAppDelegate.getInstance().run();
}, { passive: true });

window.addEventListener('beforeunload',
  (): void => LAppDelegate.releaseInstance(),
  { passive: true }
);
