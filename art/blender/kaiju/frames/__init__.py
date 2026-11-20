"""One module per frame: kaiju/frames/<frame_id>.py exposing `build(K)`.

Everything is authored in GAME coordinates: +Y up, +Z forward (the creature faces +Z), +X right,
ground at y=0. A kaiju is ~9-11 units tall (see js/render/kaiju-model.js for the legacy proportions).
"L" parts are at -X, "R" at +X (same as the JS rig).

Module-level knobs (all optional)
  DETAIL      tileable detail normal map for body/belly: 'scales' | 'skin' | 'rock' | 'chitin' | 'fur' |
              'metal' | 'cloth' | 'smooth' (none)
  UV_SCALE    detail tiling density (0.75 = default look on a 10-unit kaiju)
  METAL       True -> metallic body/belly/plate (mecha)
  EYE_ACCENT  True -> eyes glow in the accent colour instead of amber
  SCALE       model-units -> exported units. Tiny wild frames model at ~9 units tall with SCALE = 0.1
  T3_SCALE    preview scale of the tier-3 render (default 1.36 = HP_SIZE[3])
  SCALE_REF   True -> also render <frame>_scale.png next to a saurian silhouette
  CAM         dict(az=35, el=9) camera azimuth / elevation for the preview

Builder API (K)  -- see kaiju/builder.py for full signatures
  B = K.sdf()                          new SDF program (kaiju/sdf.py)
    B.tube(pts, radii, k=0.25, s=(1,1,1), mirror=False, cut=False)   spline tube, per-axis squash `s`
    B.ell(c, r, rot=None, k=0.25, mirror=False, cut=False)           ellipsoid (rot = XYZ euler degrees)
    B.box(c, half, rot=None, rad=0.1, k=0.1, mirror=False, cut=False)
    B.add(prim, k) / B.cut(prim, k)    any prim from kaiju.sdf (Ell, Box, Tube, Shell, Inter)
    B.disp(fn, band=0.16)              fn(x, y, z)->offset (negative = outward), near-surface sculpt detail
    k = smooth-blend radius (0 = hard union). mirror=True also adds the X-mirrored copy.
  K.skin(B, name='body', voxel=0.05, tris=20000, belly=maskSDF, regions=[(mat, mask)], tint=fn(P, N),
         main=True, base_mat='body', custom_normals=True, sharp=None)
      meshes + decimates the SDF, cuts clean material regions (mask = SDF or fn(P)->neg inside),
      bakes AO/tint into vertex colours. The first/main skin is what K.snap / K.plate use.
  K.snap(P) -> (points, normals)       project point(s) onto the main skin surface
  K.bone(name, head, tail, parent, r)  / K.chain(prefix, pts, parent, (r0, r1))   skeleton (r ~ body radius there)
  K.mount(name, pos, bone)             attach empties: head jaw mouth chest back tailTip handL handR
                                       shoulderL shoulderR crown
  Rigid pieces -- first arg is the destination object:
      'body' (joined into the skin) | 'pow_t0'..'pow_t3' (EXCLUSIVE sets) |
      'arm_t1..3', 'enr_t1..3', 'spd_t1..3', 'fcs_t1..3' (ADDITIVE) | 'vis_<key>'
    K.horn(part, pts, radii, mat='horn', sides, per, flat, bone)      curved spike/claw/tooth/horn
    K.spike(part, base, n, length, r)  K.claw(part, base, dir, length, r, curl)
    K.crystal(part, base, n, h)        faceted accent crystal cluster
    K.eye(part, c, gaze, r)            glowing eyeball + pupil + socket
    K.ball(part, c, r, mat)            ellipsoid
    K.leaf(part, base, n, t, h, w, core=False)   jagged dorsal plate (+ glow core variant)
    K.frill(part, base, n, t, h, w, rays)        spined fin / frill
    K.fin(part, base, n, t, h, w)                plain membrane fin
    K.plate(part, c, r, lames=3)       armour that hugs the body (offset shell of the skin SDF)
    K.sdf_piece(part, sdf, voxel, tris, mat)     any SDF as a rigid piece
    K.piece(part, V, T, mat, smooth, col, bone)  raw triangles (kaiju/geom.py has sweep/grid/strip/sphere)
  K.mat(name, color='#hex'|rgb, rough=, metal=, emit=, emit_strength=, vcol=True, detail=True,
        detail_kind=, double=True)     define / override a material slot BEFORE its first use
  Material slots: body, belly, accent (player colours) + plate, horn, dark, eye, mouth, membrane.
"""
