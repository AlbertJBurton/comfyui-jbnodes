// GLSL Shader Script to Simulate Film Grain
// -----------------------------------------
// Copyright (C) 2026  Albert J. Burton
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#version 330

// --- REQUIRED COMFYUI DECLARATIONS ---
in vec2 uv;
out vec4 FragColor;
uniform sampler2D image;

// --- CUSTOM UNIFORMS ---
uniform int iso; // ISO Scale 
uniform float morphological_variance; // Morphological Variance 
uniform float luminance_peak_bias; // Luminance Peak Bias 
uniform float signal_noise_ratio; // signal_noise Ratio 
uniform float temporal_entropy; // Temporal Entropy 
uniform float film_grit; // Film Grit
uniform float halation; // Halation Strength
uniform int grain_type;     // Emulsion Type (0 = Cubic, 1 = Tabular)
uniform int algorithmic_octaves;     // Algorithmic Octaves 
uniform int blend;     // Blend Mode (0=Soft Light, 1=Overlay, 2=Linear Light)
uniform float emulsion_softness; // Emulsion Softness (Input 0.0-2.0)
uniform float film_width; // Width of film (based on film format)

// --- EMULSION SOFTNESS (IRRADIATION) ---
vec3 sampleEmulsion(sampler2D tex, vec2 uv, float softness) {
    // Bypass blur if set to 0 for performance
    if (softness <= 0.0) {
        return texture(tex, uv).rgb;
    }
    
    vec2 texSize = vec2(textureSize(tex, 0));
    vec2 texel = 1.0 / texSize;
    vec3 color = vec3(0.0);
    float weightSum = 0.0;
    
    // Make softness resolution-independent (e.g., softness 1.0 = 0.1% of image width blur)
    float blurRadius = softness * (texSize.x * 0.001);
    
    // 5x5 Micro-Gaussian Blur
    for (float x = -2.0; x <= 2.0; x += 1.0) {
        for (float y = -2.0; y <= 2.0; y += 1.0) {
            vec2 offset = vec2(x, y) * texel * blurRadius;
            float w = exp(-(x*x + y*y) / 2.0); 
            color += texture(tex, uv + offset).rgb * w;
            weightSum += w;
        }
    }
    return color / weightSum;
}

// --- HALATION (LIGHT PIPING) ---
vec3 sampleHalation(sampler2D tex, vec2 uv, float strength, float formatMultiplier) {
    if (strength <= 0.0) return vec3(0.0);
    
    vec2 texSize = vec2(textureSize(tex, 0));
    vec2 texel = 1.0 / texSize;
    vec3 halo = vec3(0.0);
    float weightSum = 0.0;
    
    // Halation needs a fairly large radius to be visible.
    // We scale it by formatMultiplier (smaller film = relatively larger halation)
    // AND we scale it by the image resolution (texSize.x) so the effect is consistent
    // regardless of the input image size. 0.02 means 2% of the image width for 35mm.
    float radius = texSize.x * 0.02 * formatMultiplier; 
    float goldenAngle = 2.39996323;
    
    // 32-tap sparse spiral blur for a smooth, broad glow
    for (int i = 0; i < 32; i++) {
        // Normalize r so the max radius is actually 'radius'
        float r = sqrt(float(i) + 0.5) * (radius / 5.65685); 
        float theta = float(i) * goldenAngle;
        vec2 offset = vec2(cos(theta), sin(theta)) * r * texel;
        
        // Sample the negative image
        vec3 s_neg = texture(tex, uv + offset).rgb;
        
        // Convert to POSITIVE to easily identify highlights
        vec3 s_pos = 1.0 - s_neg;
        float luma = dot(s_pos, vec3(0.299, 0.587, 0.114));
        
        // Isolate highlights (anything brighter than 50% gray in the positive scene)
        float bright = smoothstep(0.5, 1.0, luma); 
        
        float w = exp(-float(i) / 10.0); 
        halo += s_pos * bright * w;
        weightSum += w;
    }
    
    vec3 haloColor = vec3(0.0);
    if (weightSum > 0.0) {
        haloColor = halo / weightSum;
    }
    
    // Return the POSITIVE glow
    return haloColor * strength;
}

// --- 3D SIMPLEX NOISE (For Cubic Grains) ---
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise3D(vec3 v){
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 = v - i + dot(i, C.xxx) ;
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );
    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;
    i = mod(i, 289.0 );
    vec4 p = permute( permute( permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
    float n_ = 1.0/7.0;
    vec3  ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
}

// --- ARTIFACT-FREE HASH ---
vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy + p3.yxx)*p3.zyx);
}

// --- 3D VORONOI / CELLULAR NOISE ---
vec2 voronoi3D(vec3 x) {
    vec3 n = floor(x);
    vec3 f = fract(x);
    float F1 = 8.0;
    float F2 = 8.0;
    for(int k=-1; k<=1; k++)
    for(int j=-1; j<=1; j++)
    for(int i=-1; i<=1; i++) {
        vec3 g = vec3(float(i),float(j),float(k));
        vec3 o = hash33( n + g );
        vec3 r = g - f + o;
        float d = dot(r,r);
        if( d < F1 ) {
            F2 = F1;
            F1 = d;
        } else if( d < F2 ) {
            F2 = d;
        }
    }
    return vec2(sqrt(F1), sqrt(F2));
}

// --- BLEND MODES ---
float blendMode(float base, float blend, int mode) {
    if (mode == 1) {
        // Overlay
        return (base < 0.5) ? 2.0 * base * blend : 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
    } else if (mode == 2) {
        // Linear Light
        return clamp(base + 2.0 * blend - 1.0, 0.0, 1.0);
    } else {
        // Soft Light (Default)
        return (blend < 0.5) 
            ? base - (1.0 - 2.0 * blend) * base * (1.0 - base) 
            : base + (2.0 * blend - 1.0) * (((base <= 0.25) ? ((16.0 * base - 12.0) * base + 4.0) * base : sqrt(base)) - base);
    }
}

vec3 applyBlend(vec3 base, vec3 blend, int mode) {
    return vec3(blendMode(base.r, blend.r, mode), 
                blendMode(base.g, blend.g, mode), 
                blendMode(base.b, blend.b, mode));
}

void main() {
    vec2 res = vec2(textureSize(image, 0));
    
    // --- FILM FORMAT SCALING ---
    // Multiplier relative to the 35mm baseline
    float formatMultiplier = 36.0 / film_width;
    
    // Step 1: Coordinate Normalization
    float safe_iso = max(float(iso), 1.0); 
    
    // Base grain size as a fraction of image width (e.g., 0.2% of width for ISO 100, 35mm)
    // This makes the grain size resolution-independent.
    float baseGrainUV = 0.002;
    
    // Scale the grain size down as the physical format gets larger
    float scaleFactorUV = sqrt(safe_iso / 100.0) * baseGrainUV * formatMultiplier;
    
    // Calculate aspect ratio to keep grains square
    float aspect = res.y / res.x;
    vec2 uv_aspect = uv * vec2(1.0, aspect);
    
    vec3 uv_noise = vec3(uv_aspect / scaleFactorUV, temporal_entropy);

    // Step 2: Base Luminance Extraction & Emulsion Softness
    // Scale softness by format: larger formats have less relative emulsion blur per-pixel
    float softness = emulsion_softness * formatMultiplier;
    vec4 originalColor = texture(image, uv);
    vec3 blurredRGB = sampleEmulsion(image, uv, softness);
    
    // Step 2.5: Add Halation
    // We sample halation from the ORIGINAL image, not the blurred one, to keep the glow sharp.
    // The sampleHalation function now returns a POSITIVE glow.
    vec3 positiveGlow = sampleHalation(image, uv, halation, formatMultiplier);
    
    // Convert our current negative blurred image to positive
    vec3 positiveBase = 1.0 - blurredRGB;
    
    // Add the glow using a Screen blend mode to prevent clipping
    vec3 positiveFinal = 1.0 - (1.0 - positiveBase) * (1.0 - positiveGlow);
    
    // Convert back to negative for the rest of the pipeline
    blurredRGB = 1.0 - positiveFinal;
    
    vec4 baseColor = vec4(blurredRGB, originalColor.a); // Preserve original alpha
    
    float L = 0.299 * baseColor.r + 0.587 * baseColor.g + 0.114 * baseColor.b;

    // Step 3: Luminance-Dependent Attenuation Masking
    float denom = max(luminance_peak_bias, 1.0 - luminance_peak_bias);
    denom = max(denom, 0.0001);
    float M = 1.0 - pow((L - luminance_peak_bias) / denom, 2.0);
    M = clamp(M, 0.0, 1.0);
    M *= smoothstep(1.0, 0.8, L); 

    // Step 4: Emulsion Morphology Generation
    float n_raw = 0.0;
    float amp = 1.0;
    float max_amp = 0.0;
    vec3 p = uv_noise;
    mat2 m2 = mat2(0.8, -0.6, 0.6, 0.8);

    if (grain_type == 0) {
        // Cubic Protocol (Simplex)
        for(int i = 0; i < 4; i++) {
            if(i >= algorithmic_octaves) break;
            n_raw += snoise3D(p) * amp;
            max_amp += amp;
            p.xy = m2 * p.xy; 
            p *= 2.0;         
            amp *= morphological_variance;  
        }
        max_amp = max(max_amp, 0.0001);
        n_raw = (n_raw / max_amp) * 0.5 + 0.5;
        
        // Map film_grit (0.0 to 1.0) to a spread factor.
        // 0.0 = softest (spread 0.5 -> bounds 0.0, 1.0)
        // 1.0 = harshest (spread 0.01 -> bounds 0.49, 0.51)
        float spread = mix(0.5, 0.01, film_grit);
        n_raw = smoothstep(0.5 - spread, 0.5 + spread, n_raw);
    } else {
        // Tabular Protocol (Voronoi)
        for(int i = 0; i < 4; i++) {
            if(i >= algorithmic_octaves) break;
            vec2 v = voronoi3D(p);
            n_raw += (v.y - v.x) * amp; 
            max_amp += amp;
            p.xy = m2 * p.xy; 
            p *= 2.0;         
            amp *= morphological_variance;  
        }
        max_amp = max(max_amp, 0.0001);
        n_raw = n_raw / max_amp; 
        
        // LUMINOSITY FIX: Center the Tabular noise
        n_raw = (n_raw - 0.25) * 2.0 + 0.5; 
    }

    // Step 5: Integration and Photometric Blending
    float n_final = (n_raw - 0.5) * M * signal_noise_ratio;
    
    vec3 grainBlend = clamp(vec3(n_final + 0.5), 0.0, 1.0);
    vec3 finalColor = applyBlend(baseColor.rgb, grainBlend, blend);

    FragColor = vec4(finalColor, baseColor.a);
}
