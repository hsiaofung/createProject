var RingShaderDict = {
  vertexShader: [
    "precision highp float;",
    "precision highp int;",
    "#define STANDARD ",
    "#define VERTEX_TEXTURES",
    "#define GAMMA_FACTOR 2",
    "#define MAX_BONES 0",
    "#define USE_ENVMAP",
    "#define ENVMAP_MODE_REFLECTION",
    "#define USE_BUMPMAP",
    "#define BONE_TEXTURE",
    "#ifdef USE_COLOR",
    "	attribute vec3 color;",
    "#endif",
    "#define PHYSICAL",
    "varying vec3 vViewPosition;",
    "#ifndef FLAT_SHADED",
    "	varying vec3 vNormal;",
    "#endif",
    "#define PI 3.14159265359",
    "#define PI2 6.28318530718",
    "#define PI_HALF 1.5707963267949",
    "#define RECIPROCAL_PI 0.31830988618",
    "#define RECIPROCAL_PI2 0.15915494",
    "#define LOG2 1.442695",
    "#define EPSILON 1e-6",
    "#define saturate(a) clamp( a, 0.0, 1.0 )",
    "#define whiteCompliment(a) ( 1.0 - saturate( a ) )",
    "float pow2( const in float x ) { return x*x; }",
    "float pow3( const in float x ) { return x*x*x; }",
    "float pow4( const in float x ) { float x2 = x*x; return x2*x2; }",
    "float average( const in vec3 color ) { return dot( color, vec3( 0.3333 ) ); }",
    "highp float rand( const in vec2 uv ) {",
    "	const highp float a = 12.9898, b = 78.233, c = 43758.5453;",
    "	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );",
    "	return fract(sin(sn) * c);",
    "}",
    "struct IncidentLight {",
    "	vec3 color;",
    "	vec3 direction;",
    "	bool visible;",
    "};",
    "struct ReflectedLight {",
    "	vec3 directDiffuse;",
    "	vec3 directSpecular;",
    "	vec3 indirectDiffuse;",
    "	vec3 indirectSpecular;",
    "};",
    "struct GeometricContext {",
    "	vec3 position;",
    "	vec3 normal;",
    "	vec3 viewDir;",
    "};",
    "vec3 transformDirection( in vec3 dir, in mat4 matrix ) {",
    "	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );",
    "}",
    "vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {",
    "	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );",
    "}",
    "vec3 projectOnPlane(in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {",
    "	float distance = dot( planeNormal, point - pointOnPlane );",
    "	return - distance * planeNormal + point;",
    "}",
    "float sideOfPlane( in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {",
    "	return sign( dot( point - pointOnPlane, planeNormal ) );",
    "}",
    "vec3 linePlaneIntersect( in vec3 pointOnLine, in vec3 lineDirection, in vec3 pointOnPlane, in vec3 planeNormal ) {",
    "	return lineDirection * ( dot( planeNormal, pointOnPlane - pointOnLine ) / dot( planeNormal, lineDirection ) ) + pointOnLine;",
    "}",
    "mat3 transposeMat3( const in mat3 m ) {",
    "	mat3 tmp;",
    "	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );",
    "	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );",
    "	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );",
    "	return tmp;",
    "}",
    "float linearToRelativeLuminance( const in vec3 color ) {",
    "	vec3 weights = vec3( 0.2126, 0.7152, 0.0722 );",
    "	return dot( weights, color.rgb );",
    "}",
    "#if defined( USE_MAP ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( USE_SPECULARMAP ) || defined( USE_ALPHAMAP ) || defined( USE_EMISSIVEMAP ) || defined( USE_ROUGHNESSMAP ) || defined( USE_METALNESSMAP )",
    "	varying vec2 vUv;",
    "	varying vec2 vUv0;",
    "	uniform mat3 uvTransform;",
    "	uniform sampler2D bumpMap;",
    "	uniform float bumpScale;	",
    "	uniform int uSurfaceType; ",
    "	uniform float displaceArea;",
    "	uniform float uHoleRadius;",
    "	uniform vec2 uHoleCenter;",
    "#endif",
    "#if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )",
    "	attribute vec2 uv2;",
    "	varying vec2 vUv2;",
    "#endif",
    "#ifdef USE_DISPLACEMENTMAP",
    "	uniform sampler2D displacementMap;",
    "	uniform float displacementScale;",
    "	uniform float displacementBias;",
    "#endif",
    "#ifdef USE_COLOR",
    "	varying vec3 vColor;",
    "#endif",
    "#ifdef USE_FOG",
    "  varying float fogDepth;",
    "#endif",
    "#ifdef USE_SHADOWMAP",
    "	#if 0 > 0",
    "		uniform mat4 directionalShadowMatrix[ 0 ];",
    "		varying vec4 vDirectionalShadowCoord[ 0 ];",
    "	#endif",
    "	#if 0 > 0",
    "		uniform mat4 spotShadowMatrix[ 0 ];",
    "		varying vec4 vSpotShadowCoord[ 0 ];",
    "	#endif",
    "	#if 3 > 0",
    "		uniform mat4 pointShadowMatrix[ 3 ];",
    "		varying vec4 vPointShadowCoord[ 3 ];",
    "	#endif",
    "#endif",
    "#ifdef USE_LOGDEPTHBUF",
    "	#ifdef USE_LOGDEPTHBUF_EXT",
    "		varying float vFragDepth;",
    "	#endif",
    "	uniform float logDepthBufFC;",
    "#endif",
    "#if 0 > 0 && ! defined( PHYSICAL ) && ! defined( PHONG )",
    "	varying vec3 vViewPosition;",
    "#endif",
    "void main() {",
    "#if defined( USE_MAP ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( USE_SPECULARMAP ) || defined( USE_ALPHAMAP ) || defined( USE_EMISSIVEMAP ) || defined( USE_ROUGHNESSMAP ) || defined( USE_METALNESSMAP )",
    "	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;",
    "	vUv0 = uv;",
    "#endif",
    "#if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )",
    "	vUv2 = uv2;",
    "#endif",
    "#ifdef USE_COLOR",
    "	vColor.xyz = color.xyz;",
    "#endif",
    "vec3 objectNormal = vec3( normal );",
    "vec3 transformedNormal = normalMatrix * objectNormal;",
    "#ifdef FLIP_SIDED",
    "	transformedNormal = - transformedNormal;",
    "#endif",
    "#ifndef FLAT_SHADED",
    "	vNormal = normalize( transformedNormal );",
    "#endif",
    "vec3 transformed = vec3( position );",
    "if(vUv0.x > 0.0){",
    "	if(uSurfaceType==3)",
    "	{",
    "		transformed += normalize( objectNormal ) * ((texture2D( bumpMap, vUv ).x-1.0) * bumpScale*0.3 + 0.0 );",
    "	}",
    "	else if(uSurfaceType==1)",
    "	{",
    "		float sc = 1.0;",
    "		float distL = length(vUv0-uHoleCenter);",
    "		if((distL < uHoleRadius+0.01) || abs(vUv0.y) > displaceArea || vUv0.x < 0.01 || vUv0.x > 1.0 - 0.01) {",
    "			sc = 0.0;",
    "		}",
    "		transformed += normalize( objectNormal ) * ((texture2D( bumpMap, vUv ).x) * bumpScale*1.5*sc + 0.0);		",
    "	}",
    "}",
    "vec4 mvPosition = modelViewMatrix * vec4( transformed, 1.0 );",
    "gl_Position = projectionMatrix * mvPosition;",
    "#ifdef USE_LOGDEPTHBUF",
    "	#ifdef USE_LOGDEPTHBUF_EXT",
    "		vFragDepth = 1.0 + gl_Position.w;",
    "	#else",
    "		gl_Position.z = log2( max( EPSILON, gl_Position.w + 1.0 ) ) * logDepthBufFC - 1.0;",
    "		gl_Position.z *= gl_Position.w;",
    "	#endif",
    "#endif",
    "#if 0 > 0 && ! defined( PHYSICAL ) && ! defined( PHONG )",
    "	vViewPosition = - mvPosition.xyz;",
    "#endif",
    "	vViewPosition = - mvPosition.xyz;",
    "#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP )",
    "	vec4 worldPosition = modelMatrix * vec4( transformed, 1.0 );",
    "#endif",
    "#ifdef USE_SHADOWMAP",
    "	#if 0 > 0",
    "	",
    "	#endif",
    "	#if 0 > 0",
    "	",
    "	#endif",
    "	#if 3 > 0",
    "	",
    "		vPointShadowCoord[ 0 ] = pointShadowMatrix[ 0 ] * worldPosition;",
    "	",
    "		vPointShadowCoord[ 1 ] = pointShadowMatrix[ 1 ] * worldPosition;",
    "	",
    "		vPointShadowCoord[ 2 ] = pointShadowMatrix[ 2 ] * worldPosition;",
    "	",
    "	#endif",
    "#endif",
    "#ifdef USE_FOG",
    "fogDepth = -mvPosition.z;",
    "#endif",
    "}",
  ],
  fragmentShader: [
    "precision highp float;",
    "precision highp int;",
    "#define STANDARD ",
    "#define GAMMA_FACTOR 2",
    "#define USE_ENVMAP",
    "#define ENVMAP_TYPE_SPHERE",
    "#define ENVMAP_MODE_REFLECTION",
    "#define ENVMAP_BLENDING_MULTIPLY",
    "#define USE_BUMPMAP",
    "// #define TEXTURE_LOD_EXT",
    "#define TONE_MAPPING",
    "#ifndef saturate",
    "	#define saturate(a) clamp( a, 0.0, 1.0 )",
    "#endif",
    "#define PHYSICAL",
    "uniform vec3 diffuse;",
    "uniform vec3 emissive;",
    "uniform float roughness;",
    "uniform float metalness;",
    "uniform float opacity;",
    "varying vec3 vViewPosition;",
    "#ifndef FLAT_SHADED",
    "	varying vec3 vNormal;",
    "#endif",
    "#define PI 3.14159265359",
    "#define PI2 6.28318530718",
    "#define PI_HALF 1.5707963267949",
    "#define RECIPROCAL_PI 0.31830988618",
    "#define RECIPROCAL_PI2 0.15915494",
    "#define LOG2 1.442695",
    "#define EPSILON 1e-6",
    "#define saturate(a) clamp( a, 0.0, 1.0 )",
    "#define whiteCompliment(a) ( 1.0 - saturate( a ) )",
    "float pow2( const in float x ) { return x*x; }",
    "float pow3( const in float x ) { return x*x*x; }",
    "float pow4( const in float x ) { float x2 = x*x; return x2*x2; }",
    "float average( const in vec3 color ) { return dot( color, vec3( 0.3333 ) ); }",
    "highp float rand( const in vec2 uv ) {",
    "	const highp float a = 12.9898, b = 78.233, c = 43758.5453;",
    "	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );",
    "	return fract(sin(sn) * c);",
    "}",
    "struct IncidentLight {",
    "	vec3 color;",
    "	vec3 direction;",
    "	bool visible;",
    "};",
    "struct ReflectedLight {",
    "	vec3 directDiffuse;",
    "	vec3 directSpecular;",
    "	vec3 indirectDiffuse;",
    "	vec3 indirectSpecular;",
    "};",
    "struct GeometricContext {",
    "	vec3 position;",
    "	vec3 normal;",
    "	vec3 viewDir;",
    "};",
    "vec3 transformDirection( in vec3 dir, in mat4 matrix ) {",
    "	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );",
    "}",
    "vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {",
    "	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );",
    "}",
    "vec3 projectOnPlane(in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {",
    "	float distance = dot( planeNormal, point - pointOnPlane );",
    "	return - distance * planeNormal + point;",
    "}",
    "float sideOfPlane( in vec3 point, in vec3 pointOnPlane, in vec3 planeNormal ) {",
    "	return sign( dot( point - pointOnPlane, planeNormal ) );",
    "}",
    "vec3 linePlaneIntersect( in vec3 pointOnLine, in vec3 lineDirection, in vec3 pointOnPlane, in vec3 planeNormal ) {",
    "	return lineDirection * ( dot( planeNormal, pointOnPlane - pointOnLine ) / dot( planeNormal, lineDirection ) ) + pointOnLine;",
    "}",
    "mat3 transposeMat3( const in mat3 m ) {",
    "	mat3 tmp;",
    "	tmp[ 0 ] = vec3( m[ 0 ].x, m[ 1 ].x, m[ 2 ].x );",
    "	tmp[ 1 ] = vec3( m[ 0 ].y, m[ 1 ].y, m[ 2 ].y );",
    "	tmp[ 2 ] = vec3( m[ 0 ].z, m[ 1 ].z, m[ 2 ].z );",
    "	return tmp;",
    "}",
    "float linearToRelativeLuminance( const in vec3 color ) {",
    "	vec3 weights = vec3( 0.2126, 0.7152, 0.0722 );",
    "	return dot( weights, color.rgb );",
    "}",
    "vec3 packNormalToRGB( const in vec3 normal ) {",
    "	return normalize( normal ) * 0.5 + 0.5;",
    "}",
    "vec3 unpackRGBToNormal( const in vec3 rgb ) {",
    "	return 2.0 * rgb.xyz - 1.0;",
    "}",
    "const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;",
    "const vec3 PackFactors = vec3( 256. * 256. * 256., 256. * 256.,  256. );",
    "const vec4 UnpackFactors = UnpackDownscale / vec4( PackFactors, 1. );",
    "const float ShiftRight8 = 1. / 256.;",
    "vec4 packDepthToRGBA( const in float v ) {",
    "	vec4 r = vec4( fract( v * PackFactors ), v );",
    "	r.yzw -= r.xyz * ShiftRight8;	return r * PackUpscale;",
    "}",
    "float unpackRGBAToDepth( const in vec4 v ) {",
    "	return dot( v, UnpackFactors );",
    "}",
    "float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {",
    "	return ( viewZ + near ) / ( near - far );",
    "}",
    "float orthographicDepthToViewZ( const in float linearClipZ, const in float near, const in float far ) {",
    "	return linearClipZ * ( near - far ) - near;",
    "}",
    "float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {",
    "	return (( near + viewZ ) * far ) / (( far - near ) * viewZ );",
    "}",
    "float perspectiveDepthToViewZ( const in float invClipZ, const in float near, const in float far ) {",
    "	return ( near * far ) / ( ( far - near ) * invClipZ - far );",
    "}",
    "#if defined( DITHERING )",
    "	vec3 dithering( vec3 color ) {",
    "		float grid_position = rand( gl_FragCoord.xy );",
    "		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );",
    "		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );",
    "		return color + dither_shift_RGB;",
    "	}",
    "#endif",
    "#ifdef USE_COLOR",
    "	varying vec3 vColor;",
    "#endif",
    "#if defined( USE_MAP ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( USE_SPECULARMAP ) || defined( USE_ALPHAMAP ) || defined( USE_EMISSIVEMAP ) || defined( USE_ROUGHNESSMAP ) || defined( USE_METALNESSMAP )",
    "	varying vec2 vUv;",
    "#endif",
    "#if defined( USE_LIGHTMAP ) || defined( USE_AOMAP )",
    "	varying vec2 vUv2;",
    "#endif",
    "uniform sampler2D logoMap;",
    "uniform float logoSc;",
    "uniform float logoSize;",
    "uniform float logoLeft;",
    "uniform vec2 logoOffset;",
    "#ifdef USE_MAP",
    "	uniform sampler2D map;",
    "#endif",
    "#ifdef USE_ALPHAMAP",
    "	uniform sampler2D alphaMap;",
    "#endif",
    "#ifdef USE_AOMAP",
    "	uniform sampler2D aoMap;",
    "	uniform float aoMapIntensity;",
    "#endif",
    "#ifdef USE_LIGHTMAP",
    "	uniform sampler2D lightMap;",
    "	uniform float lightMapIntensity;",
    "#endif",
    "#ifdef USE_EMISSIVEMAP",
    "	uniform sampler2D emissiveMap;",
    "#endif",
    "#if defined( USE_ENVMAP ) || defined( PHYSICAL )",
    "	uniform float reflectivity;",
    "	uniform float envMapIntensity;",
    "#endif",
    "#ifdef USE_ENVMAP",
    "	#if ! defined( PHYSICAL ) && ( defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) )",
    "		varying vec3 vWorldPosition;",
    "	#endif",
    "	#ifdef ENVMAP_TYPE_CUBE",
    "		uniform samplerCube envMap;",
    "	#else",
    "		uniform sampler2D envMap;",
    "	#endif",
    "	uniform float flipEnvMap;",
    "	uniform int maxMipLevel;",
    "	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( PHYSICAL )",
    "		uniform float refractionRatio;",
    "	#else",
    "		varying vec3 vReflect;",
    "	#endif",
    "#endif",
    "float punctualLightIntensityToIrradianceFactor( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {",
    "	if( decayExponent > 0.0 ) {",
    "#if defined ( PHYSICALLY_CORRECT_LIGHTS )",
    "		float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );",
    "		float maxDistanceCutoffFactor = pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );",
    "		return distanceFalloff * maxDistanceCutoffFactor;",
    "#else",
    "		return pow( saturate( -lightDistance / cutoffDistance + 1.0 ), decayExponent );",
    "#endif",
    "	}",
    "	return 1.0;",
    "}",
    "vec3 BRDF_Diffuse_Lambert( const in vec3 diffuseColor ) {",
    "	return RECIPROCAL_PI * diffuseColor;",
    "}",
    "vec3 F_Schlick( const in vec3 specularColor, const in float dotLH ) {",
    "	float fresnel = exp2( ( -5.55473 * dotLH - 6.98316 ) * dotLH );",
    "	return ( 1.0 - specularColor ) * fresnel + specularColor;",
    "}",
    "float G_GGX_Smith( const in float alpha, const in float dotNL, const in float dotNV ) {",
    "	float a2 = pow2( alpha );",
    "	float gl = dotNL + sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );",
    "	float gv = dotNV + sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );",
    "	return 1.0 / ( gl * gv );",
    "}",
    "float G_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {",
    "	float a2 = pow2( alpha );",
    "	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );",
    "	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );",
    "	return 0.5 / max( gv + gl, EPSILON );",
    "}",
    "float D_GGX( const in float alpha, const in float dotNH ) {",
    "	float a2 = pow2( alpha );",
    "	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;",
    "	return RECIPROCAL_PI * a2 / pow2( denom );",
    "}",
    "vec3 BRDF_Specular_GGX( const in IncidentLight incidentLight, const in GeometricContext geometry, const in vec3 specularColor, const in float roughness ) {",
    "	float alpha = pow2( roughness );",
    "	vec3 halfDir = normalize( incidentLight.direction + geometry.viewDir );",
    "	float dotNL = saturate( dot( geometry.normal, incidentLight.direction ) );",
    "	float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );",
    "	float dotNH = saturate( dot( geometry.normal, halfDir ) );",
    "	float dotLH = saturate( dot( incidentLight.direction, halfDir ) );",
    "	vec3 F = F_Schlick( specularColor, dotLH );",
    "	float G = G_GGX_SmithCorrelated( alpha, dotNL, dotNV );",
    "	float D = D_GGX( alpha, dotNH );",
    "	return F * ( G * D );",
    "}",
    "vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {",
    "	const float LUT_SIZE  = 64.0;",
    "	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;",
    "	const float LUT_BIAS  = 0.5 / LUT_SIZE;",
    "	float dotNV = saturate( dot( N, V ) );",
    "	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );",
    "	uv = uv * LUT_SCALE + LUT_BIAS;",
    "	return uv;",
    "}",
    "float LTC_ClippedSphereFormFactor( const in vec3 f ) {",
    "	float l = length( f );",
    "	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );",
    "}",
    "vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {",
    "	float x = dot( v1, v2 );",
    "	float y = abs( x );",
    "	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;",
    "	float b = 3.4175940 + ( 4.1616724 + y ) * y;",
    "	float v = a / b;",
    "	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;",
    "	return cross( v1, v2 ) * theta_sintheta;",
    "}",
    "vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {",
    "	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];",
    "	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];",
    "	vec3 lightNormal = cross( v1, v2 );",
    "	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );",
    "	vec3 T1, T2;",
    "	T1 = normalize( V - N * dot( V, N ) );",
    "	T2 = - cross( N, T1 );",
    "	mat3 mat = mInv * transposeMat3( mat3( T1, T2, N ) );",
    "	vec3 coords[ 4 ];",
    "	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );",
    "	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );",
    "	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );",
    "	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );",
    "	coords[ 0 ] = normalize( coords[ 0 ] );",
    "	coords[ 1 ] = normalize( coords[ 1 ] );",
    "	coords[ 2 ] = normalize( coords[ 2 ] );",
    "	coords[ 3 ] = normalize( coords[ 3 ] );",
    "	vec3 vectorFormFactor = vec3( 0.0 );",
    "	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );",
    "	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );",
    "	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );",
    "	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );",
    "	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );",
    "	return vec3( result );",
    "}",
    "vec3 BRDF_Specular_GGX_Environment( const in GeometricContext geometry, const in vec3 specularColor, const in float roughness ) {",
    "	float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );",
    "	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );",
    "	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );",
    "	vec4 r = roughness * c0 + c1;",
    "	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;",
    "	vec2 AB = vec2( -1.04, 1.04 ) * a004 + r.zw;",
    "	return specularColor * AB.x + AB.y;",
    "}",
    "float G_BlinnPhong_Implicit( ) {",
    "	return 0.25;",
    "}",
    "float D_BlinnPhong( const in float shininess, const in float dotNH ) {",
    "	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );",
    "}",
    "vec3 BRDF_Specular_BlinnPhong( const in IncidentLight incidentLight, const in GeometricContext geometry, const in vec3 specularColor, const in float shininess ) {",
    "	vec3 halfDir = normalize( incidentLight.direction + geometry.viewDir );",
    "	float dotNH = saturate( dot( geometry.normal, halfDir ) );",
    "	float dotLH = saturate( dot( incidentLight.direction, halfDir ) );",
    "	vec3 F = F_Schlick( specularColor, dotLH );",
    "	float G = G_BlinnPhong_Implicit( );",
    "	float D = D_BlinnPhong( shininess, dotNH );",
    "	return F * ( G * D );",
    "}",
    "float GGXRoughnessToBlinnExponent( const in float ggxRoughness ) {",
    "	return ( 2.0 / pow2( ggxRoughness + 0.0001 ) - 2.0 );",
    "}",
    "float BlinnExponentToGGXRoughness( const in float blinnExponent ) {",
    "	return sqrt( 2.0 / ( blinnExponent + 2.0 ) );",
    "}",
    "#ifdef ENVMAP_TYPE_CUBE_UV",
    "#define cubeUV_textureSize (1024.0)",
    "int getFaceFromDirection(vec3 direction) {",
    "	vec3 absDirection = abs(direction);",
    "	int face = -1;",
    "	if( absDirection.x > absDirection.z ) {",
    "		if(absDirection.x > absDirection.y )",
    "			face = direction.x > 0.0 ? 0 : 3;",
    "		else",
    "			face = direction.y > 0.0 ? 1 : 4;",
    "	}",
    "	else {",
    "		if(absDirection.z > absDirection.y )",
    "			face = direction.z > 0.0 ? 2 : 5;",
    "		else",
    "			face = direction.y > 0.0 ? 1 : 4;",
    "	}",
    "	return face;",
    "}",
    "#define cubeUV_maxLods1  (log2(cubeUV_textureSize*0.25) - 1.0)",
    "#define cubeUV_rangeClamp (exp2((6.0 - 1.0) * 2.0))",
    "vec2 MipLevelInfo( vec3 vec, float roughnessLevel, float roughness ) {",
    "	float scale = exp2(cubeUV_maxLods1 - roughnessLevel);",
    "	float dxRoughness = dFdx(roughness);",
    "	float dyRoughness = dFdy(roughness);",
    "	vec3 dx = dFdx( vec * scale * dxRoughness );",
    "	vec3 dy = dFdy( vec * scale * dyRoughness );",
    "	float d = max( dot( dx, dx ), dot( dy, dy ) );",
    "	d = clamp(d, 1.0, cubeUV_rangeClamp);",
    "	float mipLevel = 0.5 * log2(d);",
    "	return vec2(floor(mipLevel), fract(mipLevel));",
    "}",
    "#define cubeUV_maxLods2 (log2(cubeUV_textureSize*0.25) - 2.0)",
    "#define cubeUV_rcpTextureSize (1.0 / cubeUV_textureSize)",
    "vec2 getCubeUV(vec3 direction, float roughnessLevel, float mipLevel) {",
    "	mipLevel = roughnessLevel > cubeUV_maxLods2 - 3.0 ? 0.0 : mipLevel;",
    "	float a = 16.0 * cubeUV_rcpTextureSize;",
    "	vec2 exp2_packed = exp2( vec2( roughnessLevel, mipLevel ) );",
    "	vec2 rcp_exp2_packed = vec2( 1.0 ) / exp2_packed;",
    "	float powScale = exp2_packed.x * exp2_packed.y;",
    "	float scale = rcp_exp2_packed.x * rcp_exp2_packed.y * 0.25;",
    "	float mipOffset = 0.75*(1.0 - rcp_exp2_packed.y) * rcp_exp2_packed.x;",
    "	bool bRes = mipLevel == 0.0;",
    "	scale =  bRes && (scale < a) ? a : scale;",
    "	vec3 r;",
    "	vec2 offset;",
    "	int face = getFaceFromDirection(direction);",
    "	float rcpPowScale = 1.0 / powScale;",
    "	if( face == 0) {",
    "		r = vec3(direction.x, -direction.z, direction.y);",
    "		offset = vec2(0.0+mipOffset,0.75 * rcpPowScale);",
    "		offset.y = bRes && (offset.y < 2.0*a) ? a : offset.y;",
    "	}",
    "	else if( face == 1) {",
    "		r = vec3(direction.y, direction.x, direction.z);",
    "		offset = vec2(scale+mipOffset, 0.75 * rcpPowScale);",
    "		offset.y = bRes && (offset.y < 2.0*a) ? a : offset.y;",
    "	}",
    "	else if( face == 2) {",
    "		r = vec3(direction.z, direction.x, direction.y);",
    "		offset = vec2(2.0*scale+mipOffset, 0.75 * rcpPowScale);",
    "		offset.y = bRes && (offset.y < 2.0*a) ? a : offset.y;",
    "	}",
    "	else if( face == 3) {",
    "		r = vec3(direction.x, direction.z, direction.y);",
    "		offset = vec2(0.0+mipOffset,0.5 * rcpPowScale);",
    "		offset.y = bRes && (offset.y < 2.0*a) ? 0.0 : offset.y;",
    "	}",
    "	else if( face == 4) {",
    "		r = vec3(direction.y, direction.x, -direction.z);",
    "		offset = vec2(scale+mipOffset, 0.5 * rcpPowScale);",
    "		offset.y = bRes && (offset.y < 2.0*a) ? 0.0 : offset.y;",
    "	}",
    "	else {",
    "		r = vec3(direction.z, -direction.x, direction.y);",
    "		offset = vec2(2.0*scale+mipOffset, 0.5 * rcpPowScale);",
    "		offset.y = bRes && (offset.y < 2.0*a) ? 0.0 : offset.y;",
    "	}",
    "	r = normalize(r);",
    "	float texelOffset = 0.5 * cubeUV_rcpTextureSize;",
    "	vec2 s = ( r.yz / abs( r.x ) + vec2( 1.0 ) ) * 0.5;",
    "	vec2 base = offset + vec2( texelOffset );",
    "	return base + s * ( scale - 2.0 * texelOffset );",
    "}",
    "#define cubeUV_maxLods3 (log2(cubeUV_textureSize*0.25) - 3.0)",
    "vec4 textureCubeUV(vec3 reflectedDirection, float roughness ) {",
    "	float roughnessVal = roughness* cubeUV_maxLods3;",
    "	float r1 = floor(roughnessVal);",
    "	float r2 = r1 + 1.0;",
    "	float t = fract(roughnessVal);",
    "	vec2 mipInfo = MipLevelInfo(reflectedDirection, r1, roughness);",
    "	float s = mipInfo.y;",
    "	float level0 = mipInfo.x;",
    "	float level1 = level0 + 1.0;",
    "	level1 = level1 > 5.0 ? 5.0 : level1;",
    "	level0 += min( floor( s + 0.5 ), 5.0 );",
    "	vec2 uv_10 = getCubeUV(reflectedDirection, r1, level0);",
    "	vec4 color10 = envMapTexelToLinear(texture2D(envMap, uv_10));",
    "	vec2 uv_20 = getCubeUV(reflectedDirection, r2, level0);",
    "	vec4 color20 = envMapTexelToLinear(texture2D(envMap, uv_20));",
    "	vec4 result = mix(color10, color20, t);",
    "	return vec4(result.rgb, 1.0);",
    "}",
    "#endif",
    "uniform vec3 ambientLightColor;",
    "vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {",
    "	vec3 irradiance = ambientLightColor;",
    "	#ifndef PHYSICALLY_CORRECT_LIGHTS",
    "		irradiance *= PI;",
    "	#endif",
    "	return irradiance;",
    "}",
    "#if 3 > 0",
    "	struct PointLight {",
    "		vec3 position;",
    "		vec3 color;",
    "		float distance;",
    "		float decay;",
    "		int shadow;",
    "		float shadowBias;",
    "		float shadowRadius;",
    "		vec2 shadowMapSize;",
    "		float shadowCameraNear;",
    "		float shadowCameraFar;",
    "	};",
    "	uniform PointLight pointLights[ 3 ];",
    "	void getPointDirectLightIrradiance( const in PointLight pointLight, const in GeometricContext geometry, out IncidentLight directLight ) {",
    "		vec3 lVector = pointLight.position - geometry.position;",
    "		directLight.direction = normalize( lVector );",
    "		float lightDistance = length( lVector );",
    "		directLight.color = pointLight.color;",
    "		directLight.color *= punctualLightIntensityToIrradianceFactor( lightDistance, pointLight.distance, pointLight.decay );",
    "		directLight.visible = ( directLight.color != vec3( 0.0 ) );",
    "	}",
    "#endif",
    "#if defined( USE_ENVMAP ) && defined( PHYSICAL )",
    "	vec3 getLightProbeIndirectIrradiance( const in GeometricContext geometry, const in int maxMIPLevel ) {",
    "		vec3 worldNormal = inverseTransformDirection( geometry.normal, viewMatrix );",
    "		#ifdef ENVMAP_TYPE_CUBE",
    "			vec3 queryVec = vec3( flipEnvMap * worldNormal.x, worldNormal.yz );",
    "			#ifdef TEXTURE_LOD_EXT",
    "				vec4 envMapColor = textureCubeLodEXT( envMap, queryVec, float( maxMIPLevel ) );",
    "			#else",
    "				vec4 envMapColor = textureCube( envMap, queryVec, float( maxMIPLevel ) );",
    "			#endif",
    "			envMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;",
    "		#elif defined( ENVMAP_TYPE_CUBE_UV )",
    "			vec3 queryVec = vec3( flipEnvMap * worldNormal.x, worldNormal.yz );",
    "			vec4 envMapColor = textureCubeUV( queryVec, 1.0 );",
    "		#else",
    "			vec4 envMapColor = vec4( 0.0 );",
    "		#endif",
    "		return PI * envMapColor.rgb * envMapIntensity;",
    "	}",
    "	float getSpecularMIPLevel( const in float blinnShininessExponent, const in int maxMIPLevel ) {",
    "		float maxMIPLevelScalar = float( maxMIPLevel );",
    "		float desiredMIPLevel = maxMIPLevelScalar + 0.79248 - 0.5 * log2( pow2( blinnShininessExponent ) + 1.0 );",
    "		return clamp( desiredMIPLevel, 0.0, maxMIPLevelScalar );",
    "	}",
    "	float specularLevel = 1.0;",
    "	vec3 getLightProbeIndirectRadiance( const in GeometricContext geometry, const in float blinnShininessExponent, const in int maxMIPLevel ) {",
    "		#ifdef ENVMAP_MODE_REFLECTION",
    "			vec3 reflectVec = reflect( -geometry.viewDir, geometry.normal );",
    "		#else",
    "			vec3 reflectVec = refract( -geometry.viewDir, geometry.normal, refractionRatio );",
    "		#endif",
    "		reflectVec = inverseTransformDirection( reflectVec, viewMatrix );",
    "		float specularMIPLevel = getSpecularMIPLevel( blinnShininessExponent, maxMIPLevel );",
    "		if(specularLevel > 1.0) specularMIPLevel =  specularLevel;",
    "		#ifdef ENVMAP_TYPE_CUBE",
    "			vec3 queryReflectVec = vec3( flipEnvMap * reflectVec.x, reflectVec.yz );",
    "			#ifdef TEXTURE_LOD_EXT",
    "				vec4 envMapColor = textureCubeLodEXT( envMap, queryReflectVec, specularMIPLevel );",
    "			#else",
    "				vec4 envMapColor = textureCube( envMap, queryReflectVec, specularMIPLevel );",
    "			#endif",
    "			envMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;",
    "		#elif defined( ENVMAP_TYPE_CUBE_UV )",
    "			vec3 queryReflectVec = vec3( flipEnvMap * reflectVec.x, reflectVec.yz );",
    "			vec4 envMapColor = textureCubeUV(queryReflectVec, BlinnExponentToGGXRoughness(blinnShininessExponent));",
    "		#elif defined( ENVMAP_TYPE_EQUIREC )",
    "			vec2 sampleUV;",
    "			sampleUV.y = asin( clamp( reflectVec.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;",
    "			sampleUV.x = atan( reflectVec.z, reflectVec.x ) * RECIPROCAL_PI2 + 0.5;",
    "			#ifdef TEXTURE_LOD_EXT",
    "				vec4 envMapColor = texture2DLodEXT( envMap, sampleUV, specularMIPLevel );",
    "			#else",
    "				vec4 envMapColor = texture2D( envMap, sampleUV, specularMIPLevel );",
    "			#endif",
    "			envMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;",
    "		#elif defined( ENVMAP_TYPE_SPHERE )",
    "			vec3 reflectView = normalize( ( viewMatrix * vec4( reflectVec, 0.0 ) ).xyz + vec3( 0.0,0.0,1.0 ) );",
    "			#ifdef TEXTURE_LOD_EXT",
    "				vec4 envMapColor = texture2DLodEXT( envMap, reflectView.xy * 0.5 + 0.5, specularMIPLevel );",
    "			#else",
    "				vec4 envMapColor = texture2D( envMap, reflectView.xy * 0.5 + 0.5,specularMIPLevel);",
    "			#endif",
    "			envMapColor.rgb = envMapTexelToLinear( envMapColor ).rgb;",
    "		#endif",
    "		return envMapColor.rgb * envMapIntensity;",
    "	}",
    "#endif",
    "struct PhysicalMaterial {",
    "	vec3	diffuseColor;",
    "	float	specularRoughness;",
    "	vec3	specularColor;",
    "	#ifndef STANDARD",
    "		float clearCoat;",
    "		float clearCoatRoughness;",
    "	#endif",
    "};",
    "#define MAXIMUM_SPECULAR_COEFFICIENT 0.16",
    "#define DEFAULT_SPECULAR_COEFFICIENT 0.04",
    "float clearCoatDHRApprox( const in float roughness, const in float dotNL ) {",
    "	return DEFAULT_SPECULAR_COEFFICIENT + ( 1.0 - DEFAULT_SPECULAR_COEFFICIENT ) * ( pow( 1.0 - dotNL, 5.0 ) * pow( 1.0 - roughness, 2.0 ) );",
    "}",
    "void RE_Direct_Physical( const in IncidentLight directLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {",
    "	float dotNL = saturate( dot( geometry.normal, directLight.direction ) );",
    "	vec3 irradiance = dotNL * directLight.color;",
    "	#ifndef PHYSICALLY_CORRECT_LIGHTS",
    "		irradiance *= PI;",
    "	#endif",
    "	#ifndef STANDARD",
    "		float clearCoatDHR = material.clearCoat * clearCoatDHRApprox( material.clearCoatRoughness, dotNL );",
    "	#else",
    "		float clearCoatDHR = 0.0;",
    "	#endif",
    "	reflectedLight.directSpecular += ( 1.0 - clearCoatDHR ) * irradiance * BRDF_Specular_GGX( directLight, geometry, material.specularColor, material.specularRoughness );",
    "	reflectedLight.directDiffuse += ( 1.0 - clearCoatDHR ) * irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );",
    "	#ifndef STANDARD",
    "		reflectedLight.directSpecular += irradiance * material.clearCoat * BRDF_Specular_GGX( directLight, geometry, vec3( DEFAULT_SPECULAR_COEFFICIENT ), material.clearCoatRoughness );",
    "	#endif",
    "}",
    "void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {",
    "	reflectedLight.indirectDiffuse += irradiance * BRDF_Diffuse_Lambert( material.diffuseColor );",
    "}",
    "void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 clearCoatRadiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {",
    "	#ifndef STANDARD",
    "		float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );",
    "		float dotNL = dotNV;",
    "		float clearCoatDHR = material.clearCoat * clearCoatDHRApprox( material.clearCoatRoughness, dotNL );",
    "	#else",
    "		float clearCoatDHR = 0.0;",
    "	#endif",
    "	reflectedLight.indirectSpecular += ( 1.0 - clearCoatDHR ) * radiance * BRDF_Specular_GGX_Environment( geometry, material.specularColor, material.specularRoughness );",
    "	#ifndef STANDARD",
    "		reflectedLight.indirectSpecular += clearCoatRadiance * material.clearCoat * BRDF_Specular_GGX_Environment( geometry, vec3( DEFAULT_SPECULAR_COEFFICIENT ), material.clearCoatRoughness );",
    "	#endif",
    "}",
    "#define RE_Direct				RE_Direct_Physical",
    "#define RE_Direct_RectArea		RE_Direct_RectArea_Physical",
    "#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical",
    "#define RE_IndirectSpecular		RE_IndirectSpecular_Physical",
    "#define Material_BlinnShininessExponent( material )   GGXRoughnessToBlinnExponent( material.specularRoughness )",
    "#define Material_ClearCoat_BlinnShininessExponent( material )   GGXRoughnessToBlinnExponent( material.clearCoatRoughness )",
    "float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {",
    "	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );",
    "}",
    "#ifdef USE_SHADOWMAP",
    "	#if 3 > 0",
    "		uniform sampler2D pointShadowMap[ 3 ];",
    "		varying vec4 vPointShadowCoord[ 3 ];",
    "	#endif",
    "	float texture2DCompare( sampler2D depths, vec2 uv, float compare ) {",
    "		return step( compare, unpackRGBAToDepth( texture2D( depths, uv ) ) );",
    "	}",
    "	float texture2DShadowLerp( sampler2D depths, vec2 size, vec2 uv, float compare ) {",
    "		const vec2 offset = vec2( 0.0, 1.0 );",
    "		vec2 texelSize = vec2( 1.0 ) / size;",
    "		vec2 centroidUV = floor( uv * size + 0.5 ) / size;",
    "		float lb = texture2DCompare( depths, centroidUV + texelSize * offset.xx, compare );",
    "		float lt = texture2DCompare( depths, centroidUV + texelSize * offset.xy, compare );",
    "		float rb = texture2DCompare( depths, centroidUV + texelSize * offset.yx, compare );",
    "		float rt = texture2DCompare( depths, centroidUV + texelSize * offset.yy, compare );",
    "		vec2 f = fract( uv * size + 0.5 );",
    "		float a = mix( lb, lt, f.y );",
    "		float b = mix( rb, rt, f.y );",
    "		float c = mix( a, b, f.x );",
    "		return c;",
    "	}",
    "	float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowBias, float shadowRadius, vec4 shadowCoord ) {",
    "		float shadow = 1.0;",
    "		shadowCoord.xyz /= shadowCoord.w;",
    "		shadowCoord.z += shadowBias;",
    "		bvec4 inFrustumVec = bvec4 ( shadowCoord.x >= 0.0, shadowCoord.x <= 1.0, shadowCoord.y >= 0.0, shadowCoord.y <= 1.0 );",
    "		bool inFrustum = all( inFrustumVec );",
    "		bvec2 frustumTestVec = bvec2( inFrustum, shadowCoord.z <= 1.0 );",
    "		bool frustumTest = all( frustumTestVec );",
    "		if ( frustumTest ) {",
    "		#if defined( SHADOWMAP_TYPE_PCF )",
    "			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;",
    "			float dx0 = - texelSize.x * shadowRadius;",
    "			float dy0 = - texelSize.y * shadowRadius;",
    "			float dx1 = + texelSize.x * shadowRadius;",
    "			float dy1 = + texelSize.y * shadowRadius;",
    "			shadow = (",
    "				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +",
    "				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +",
    "				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +",
    "				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +",
    "				texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z ) +",
    "				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +",
    "				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +",
    "				texture2DCompare( shadowMap, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +",
    "				texture2DCompare( shadowMap, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )",
    "			) * ( 1.0 / 9.0 );",
    "		#elif defined( SHADOWMAP_TYPE_PCF_SOFT )",
    "			vec2 texelSize = vec2( 1.0 ) / shadowMapSize;",
    "			float dx0 = - texelSize.x * shadowRadius;",
    "			float dy0 = - texelSize.y * shadowRadius;",
    "			float dx1 = + texelSize.x * shadowRadius;",
    "			float dy1 = + texelSize.y * shadowRadius;",
    "			shadow = (",
    "				texture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx0, dy0 ), shadowCoord.z ) +",
    "				texture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( 0.0, dy0 ), shadowCoord.z ) +",
    "				texture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx1, dy0 ), shadowCoord.z ) +",
    "				texture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx0, 0.0 ), shadowCoord.z ) +",
    "				texture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy, shadowCoord.z ) +",
    "				texture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx1, 0.0 ), shadowCoord.z ) +",
    "				texture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx0, dy1 ), shadowCoord.z ) +",
    "				texture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( 0.0, dy1 ), shadowCoord.z ) +",
    "				texture2DShadowLerp( shadowMap, shadowMapSize, shadowCoord.xy + vec2( dx1, dy1 ), shadowCoord.z )",
    "			) * ( 1.0 / 9.0 );",
    "		#else",
    "			shadow = texture2DCompare( shadowMap, shadowCoord.xy, shadowCoord.z );",
    "		#endif",
    "		}",
    "		return shadow;",
    "	}",
    "	vec2 cubeToUV( vec3 v, float texelSizeY ) {",
    "		vec3 absV = abs( v );",
    "		float scaleToCube = 1.0 / max( absV.x, max( absV.y, absV.z ) );",
    "		absV *= scaleToCube;",
    "		v *= scaleToCube * ( 1.0 - 2.0 * texelSizeY );",
    "		vec2 planar = v.xy;",
    "		float almostATexel = 1.5 * texelSizeY;",
    "		float almostOne = 1.0 - almostATexel;",
    "		if ( absV.z >= almostOne ) {",
    "			if ( v.z > 0.0 )",
    "				planar.x = 4.0 - v.x;",
    "		} else if ( absV.x >= almostOne ) {",
    "			float signX = sign( v.x );",
    "			planar.x = v.z * signX + 2.0 * signX;",
    "		} else if ( absV.y >= almostOne ) {",
    "			float signY = sign( v.y );",
    "			planar.x = v.x + 2.0 * signY + 2.0;",
    "			planar.y = v.z * signY - 2.0;",
    "		}",
    "		return vec2( 0.125, 0.25 ) * planar + vec2( 0.375, 0.75 );",
    "	}",
    "	float getPointShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {",
    "		vec2 texelSize = vec2( 1.0 ) / ( shadowMapSize * vec2( 4.0, 2.0 ) );",
    "		vec3 lightToPosition = shadowCoord.xyz;",
    "		float dp = ( length( lightToPosition ) - shadowCameraNear ) / ( shadowCameraFar - shadowCameraNear );		dp += shadowBias;",
    "		vec3 bd3D = normalize( lightToPosition );",
    "		#if defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_PCF_SOFT )",
    "			vec2 offset = vec2( - 1, 1 ) * shadowRadius * texelSize.y;",
    "			return (",
    "				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyy, texelSize.y ), dp ) +",
    "				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyy, texelSize.y ), dp ) +",
    "				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xyx, texelSize.y ), dp ) +",
    "				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yyx, texelSize.y ), dp ) +",
    "				texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp ) +",
    "				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxy, texelSize.y ), dp ) +",
    "				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxy, texelSize.y ), dp ) +",
    "				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.xxx, texelSize.y ), dp ) +",
    "				texture2DCompare( shadowMap, cubeToUV( bd3D + offset.yxx, texelSize.y ), dp )",
    "			) * ( 1.0 / 9.0 );",
    "		#else",
    "			return texture2DCompare( shadowMap, cubeToUV( bd3D, texelSize.y ), dp );",
    "		#endif",
    "	}",
    "#endif",
    "#ifdef USE_BUMPMAP",
    "	uniform sampler2D bumpMap;",
    "	uniform float bumpScale;",
    "	vec2 bumpUv;",
    "	float bumpScaleValue = 0.0;",
    "	vec2 dHdxy_fwd() {",
    "		vec2 dSTdx = dFdx( bumpUv );",
    "		vec2 dSTdy = dFdy( bumpUv );",
    "		float Hll = bumpScaleValue * texture2D( bumpMap, bumpUv ).x;",
    "		float dBx = bumpScaleValue * texture2D( bumpMap, bumpUv + dSTdx ).x - Hll;",
    "		float dBy = bumpScaleValue * texture2D( bumpMap, bumpUv + dSTdy ).x - Hll;",
    "		return vec2( dBx, dBy );",
    "	}",
    "	uniform sampler2D sandMap;",
    "	vec2 dHdxy_fwd_sand(vec2 buv, float sc) {",
    "		vec2 dSTdx = dFdx( buv );",
    "		vec2 dSTdy = dFdy( buv );",
    "		float Hll = sc * texture2D( sandMap, buv ).x;",
    "		float dBx = sc * texture2D( sandMap, buv + dSTdx ).x - Hll;",
    "		float dBy = sc * texture2D( sandMap, buv + dSTdy ).x - Hll;",
    "		return vec2( dBx, dBy );",
    "	}",
    "	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy ) {",
    "		vec3 vSigmaX = vec3( dFdx( surf_pos.x ), dFdx( surf_pos.y ), dFdx( surf_pos.z ) );",
    "		vec3 vSigmaY = vec3( dFdy( surf_pos.x ), dFdy( surf_pos.y ), dFdy( surf_pos.z ) );",
    "		vec3 vN = surf_norm;",
    "		vec3 R1 = cross( vSigmaY, vN );",
    "		vec3 R2 = cross( vN, vSigmaX );",
    "		float fDet = dot( vSigmaX, R1 );",
    "		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );",
    "		return normalize( abs( fDet ) * surf_norm - vGrad );",
    "	}",
    "#endif",
    "#ifdef USE_NORMALMAP",
    "	uniform sampler2D normalMap;",
    "	uniform vec2 normalScale;",
    "	vec3 perturbNormal2Arb( vec3 eye_pos, vec3 surf_norm ) {",
    "		vec3 q0 = vec3( dFdx( eye_pos.x ), dFdx( eye_pos.y ), dFdx( eye_pos.z ) );",
    "		vec3 q1 = vec3( dFdy( eye_pos.x ), dFdy( eye_pos.y ), dFdy( eye_pos.z ) );",
    "		vec2 st0 = dFdx( vUv.st );",
    "		vec2 st1 = dFdy( vUv.st );",
    "		float scale = sign( st1.t * st0.s - st0.t * st1.s );		scale *= float( gl_FrontFacing ) * 2.0 - 1.0;",
    "		vec3 S = normalize( ( q0 * st1.t - q1 * st0.t ) * scale );",
    "		vec3 T = normalize( ( - q0 * st1.s + q1 * st0.s ) * scale );",
    "		vec3 N = normalize( surf_norm );",
    "		vec3 mapN = texture2D( normalMap, vUv ).xyz * 2.0 - 1.0;",
    "		mapN.xy = normalScale * mapN.xy;",
    "		mat3 tsn = mat3( S, T, N );",
    "		return normalize( tsn * mapN );",
    "	}",
    "#endif",
    "#ifdef USE_ROUGHNESSMAP",
    "	uniform sampler2D roughnessMap;",
    "#endif",
    "#ifdef USE_METALNESSMAP",
    "	uniform sampler2D metalnessMap;",
    "#endif",
    "#ifdef USE_LOGDEPTHBUF",
    "	uniform float logDepthBufFC;",
    "	#ifdef USE_LOGDEPTHBUF_EXT",
    "		varying float vFragDepth;",
    "	#endif",
    "#endif",
    "#if 0 > 0",
    "	#if ! defined( PHYSICAL ) && ! defined( PHONG )",
    "		varying vec3 vViewPosition;",
    "	#endif",
    "	uniform vec4 clippingPlanes[ 0 ];",
    "#endif",
    "varying vec2 vUv0;",
    "uniform vec2 uHoleCenter;",
    "uniform float uHoleRadius; ",
    "uniform vec2 uHoleCenter1;",
    "uniform float uHoleRadius1; ",
    "uniform int uSurfaceType; ",
    "void main() {",
    "	vec4 diffuseColor = vec4( diffuse, opacity );",
    "	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );",
    "	vec3 totalEmissiveRadiance = emissive;",
    "#ifdef ALPHATEST",
    "	if ( diffuseColor.a < ALPHATEST ) discard;",
    "#endif",
    "float roughnessFactor = roughness;",
    "float metalnessFactor = metalness;",
    "float cameraDist = length(cameraPosition);",
    "vec3 normal = normalize( vNormal );",
    "normal = normal * ( float( gl_FrontFacing ) * 2.0 - 1.0 );",
    "if(vUv0.x > 0.0)",
    "{",
    "	#ifdef USE_NORMALMAP",
    "		normal = perturbNormal2Arb( -vViewPosition, normal );",
    "	#elif defined( USE_BUMPMAP )",
    "		bumpScaleValue = bumpScale;",
    "		bumpUv = vUv;",
    "		if(uSurfaceType == 2||uSurfaceType==4)",
    "		{",
    "			bumpScaleValue*= cameraDist / 35.0;",
    "		}",
    "		normal = perturbNormalArb( -vViewPosition, normal, dHdxy_fwd() );",
    "		if(uSurfaceType==1)",
    "		{",
    "			normal = perturbNormalArb(-vViewPosition,normal,dHdxy_fwd_sand(vUv*2.0,0.1));",
    "		}",
    "	#endif",
    "	if(uSurfaceType==2)",
    "	{",
    "		metalnessFactor = 0.88;",
    "		roughnessFactor = 1.0 - metalnessFactor+0.0;",
    "		specularLevel = 3.5;",
    "	}",
    "	else if(uSurfaceType==4)",
    "	{",
    "		metalnessFactor = 0.88;",
    "		roughnessFactor = 1.0 - metalnessFactor+0.0;",
    "		specularLevel = 2.0;				",
    "	}	",
    "	else if(uSurfaceType==1)",
    "	{",
    "		metalnessFactor = 0.9;",
    "		roughnessFactor = 1.0 - metalnessFactor+0.1;	",
    "		specularLevel = 2.0;					",
    "	}",
    "	else if(uSurfaceType==3)",
    "	{",
    "		metalnessFactor = 0.9;",
    "		roughnessFactor = 1.0 - metalnessFactor+0.1;	",
    "	}",
    "}",
    "#ifdef USE_EMISSIVEMAP",
    "	vec4 emissiveColor = texture2D( emissiveMap, vUv );",
    "	emissiveColor.rgb = emissiveMapTexelToLinear( emissiveColor ).rgb;",
    "	totalEmissiveRadiance *= emissiveColor.rgb;",
    "#endif",
    "	if(vUv0.x > 0.0 && uHoleRadius > 0.0)",
    "	{",
    "		float distL = length(vUv0-uHoleCenter);",
    "		if(distL < uHoleRadius)",
    "		{",
    "			discard;",
    "		}",
    "	}",
    "	if(vUv0.x < 0.0 && uHoleRadius1 > 0.0)",
    "	{",
    "		float distL = length(vUv0-uHoleCenter1);",
    "		if(distL < uHoleRadius1)",
    "		{",
    "			discard;",
    "		}	",
    "		if(uHoleCenter1.x<=-1.0)	",
    "		{",
    "			distL = length(vUv0-vec2(0.0,uHoleCenter1.y));",
    "			if(distL < uHoleRadius1)",
    "			{",
    "				discard;",
    "			}			",
    "		}",
    "	}",
    "PhysicalMaterial material;",
    "material.diffuseColor = diffuseColor.rgb * ( 1.0 - metalnessFactor );",
    "material.specularRoughness = clamp( roughnessFactor, 0.04, 1.0 );",
    "material.specularColor = mix( vec3( DEFAULT_SPECULAR_COEFFICIENT ), diffuseColor.rgb, metalnessFactor );",
    "GeometricContext geometry;",
    "geometry.position = - vViewPosition;",
    "geometry.normal = normal;",
    "geometry.viewDir = normalize( vViewPosition );",
    "IncidentLight directLight;",
    "// #if ( 3 > 0 ) && defined( RE_Direct )",
    "PointLight pointLight;",
    "// pointLight.position = transformDirection( vec3(-3.0, -10.2, 13.5),viewMatrix);",
    "// pointLight.color = vec3(0.15);",
    "// pointLight.distance = 2000.0;",
    "// pointLight.shadow = 0;",
    "// pointLight.decay = 1.0;",
    "// getPointDirectLightIrradiance( pointLight, geometry, directLight );",
    "// RE_Direct( directLight, geometry, material, reflectedLight );",
    "// pointLight.position = transformDirection(vec3(16, -4, -25),viewMatrix);",
    "// pointLight.color = vec3(0.15);",
    "// pointLight.distance = 60.0;",
    "// pointLight.shadow = 0;",
    "// pointLight.decay = 1.0;",
    "// getPointDirectLightIrradiance( pointLight, geometry, directLight );",
    "// RE_Direct( directLight, geometry, material, reflectedLight );",
    "// pointLight.position = transformDirection(vec3( -13, 3.2, -6.9),viewMatrix);",
    "// pointLight.color = vec3(0.15);",
    "// pointLight.distance = 50.0;",
    "// pointLight.shadow = 0;",
    "// pointLight.decay = 1.0;",
    "// getPointDirectLightIrradiance( pointLight, geometry, directLight );",
    "// RE_Direct( directLight, geometry, material, reflectedLight );",
    "  // vec3 outgoingLight1 = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;",
    "  // gl_FragColor =  vec4( outgoingLight1, diffuseColor.a );",
    "  // return;",
    "// #endif",
    "#if defined( RE_IndirectDiffuse )",
    "	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );",
    "#endif",
    "#if defined( RE_IndirectSpecular )",
    "	vec3 radiance = vec3( 0.0 );",
    "	vec3 clearCoatRadiance = vec3( 0.0 );",
    "#endif",
    "#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )",
    "	radiance += getLightProbeIndirectRadiance( geometry, Material_BlinnShininessExponent( material ), maxMipLevel );",
    "#endif",
    "#if defined( RE_IndirectDiffuse )",
    "	RE_IndirectDiffuse( irradiance, geometry, material, reflectedLight );",
    "#endif",
    "#if defined( RE_IndirectSpecular )",
    "	RE_IndirectSpecular( radiance, clearCoatRadiance, geometry, material, reflectedLight );",
    "#endif",
    "    vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;",
    "	if(vUv.x>0.0 && uSurfaceType==3)",
    "	{",
    "		vec4 texelColor = texture2D( bumpMap, vUv );",
    "		texelColor = mapTexelToLinear( texelColor );",
    "		// outgoingLight *= (0.7 + texelColor.r*0.3);	",
    "	}    ",
    "	gl_FragColor = vec4( outgoingLight, diffuseColor.a );",
    "  gl_FragColor = linearToOutputTexel( gl_FragColor );",
    "	if(vUv0.x < 0.0)",
    "	{	",
    "		float vx = -vUv0.x + logoOffset.x;",
    "		if(vx>1.0) vx = vx - 1.0;",
    "		if(vx<0.0) vx = 1.0 + vx;",
    "		vec2 tUv = vec2(vx*logoSc,(vUv0.y)*logoSc*logoSize+logoOffset.y);",
    "		float trow = floor(tUv.x);",
    "		float tcol = floor(tUv.y);",
    "		float try = tUv.y - tcol;",
    "		float trx = tUv.x - trow;",
    "		if(tUv.x>=logoLeft && tUv.x < logoLeft+1.0 && (tcol==0.0))",
    "		{",
    "			gl_FragColor = vec4(outgoingLight * texture2D(logoMap,vec2(tUv.x,try)).x,1.0);",
    "			return;",
    "		}",
    "	}",
    "	if(vUv0.x > 0.0 && uSurfaceType == 3)",
    "	{",
    "		float ax = texture2D(bumpMap,vUv).x + 0.1;",
    "		if(ax > 1.0) ax = 1.0;",
    "		gl_FragColor = vec4(outgoingLight * ax, 1.0);",
    "	}",
    "  // gl_FragColor = texture2D(bumpMap,vUv*13.0);",
    "#ifdef PREMULTIPLIED_ALPHA",
    "	gl_FragColor.rgb *= gl_FragColor.a;",
    "#endif",
    "}",
  ],
};

var PlaneShaderDict = {
  vertexShader: [
    "varying vec2 vUv;",
    "varying vec3 vpos;",
    "void main() {",
    "    vUv = uv;",
    "    gl_Position = projectionMatrix*modelViewMatrix * vec4(position, 1.0 );",
    "    vpos = position;",
    "}",
  ],
  fragmentShader: [
    " varying vec2 vUv;",
    " varying vec3 vpos;",
    " uniform sampler2D tex0;",
    " uniform vec3 center0;",
    " uniform vec3 axisX0;",
    " uniform vec3 axisY0;",
    " uniform vec3 center1;",
    " uniform vec3 axisX1;",
    " uniform vec3 axisY1;	 	 ",
    " uniform float width0;",
    " uniform float width1;       ",
    " void main() {	",
    " 	float distX0 = dot((vpos - center0),axisX0);",
    " 	float distY0 = dot((vpos - center0),axisY0);",
    " 	float len0 = length(vec2(distX0/2.5,distY0)) / width0;",
    " 	float distX1 = dot((vpos - center1),axisX1);",
    " 	float distY1 = dot((vpos - center1),axisY1);",
    " 	float len1 = length(vec2(distX1/2.5,distY1)) / width1;",
    " 	float len = min(len0,len1);",
    " 	//len = pow(len,4.0);",
    " 	if(len > 1.0) len = 1.0;",
    " 	//len = min(len,c0);",
    " 	gl_FragColor = vec4(vec3(0.0),1.0-len);",
    " }",
  ],
};

var DiamondColorDict = {
  aq: [[0, 0.6, 1], 0.75, 0.2],
  at: [[0.47, 0.3, 0.67], 1, 0.6],
  cr: [[0.93, 0.52, 0.19], 0.75, 0.5],
  dd: [[1.1, 1.1, 1.1], 1.25, 1],
  em: [[0.2, 1, 0.45], 1, 0.6],
  fm: [[0.84, 0.84, 0.84], 0.5, 1],
  ga: [[0.83, 0.23, 0], 0.5, 0.8],
  pe: [[0.6, 0.88, 0], 0.75, 0.3],
  ru: [[1, 0, 0.3], 1.25, 0.8],
  sa: [[0, 0.2, 1], 1.25, 0.8],
  tu: [[0.7, 0, 0.53], 0.75, 0.3],
  tz: [[0.57, 0.53, 0.9], 1, 0.5],
};
var RingModel = function() {
  this.geometry = null;
  this.base_ring = null;
  this.radius = 10;
  this.material = null;
  this.mesh = null;
  this.mirror_mesh = null;
  this.diamond_mesh = null;
  this.inner_diamond_mesh = null;
  this.isShowMirror = false;
  var scope = this;
  this.envMap = new THREE.Texture();
  this.bumpMap = new THREE.Texture();
  this.logoMap = new THREE.Texture();
  this.sandMap = new THREE.Texture();
  this.diffuseColor = new THREE.Color().setRGB(1.354, 1.3666, 1.37);
  this.position = new THREE.Vector3();
  this.rotation = new THREE.Euler();
  this.shapeType = "p";
  this.widthType = "35";
  this.colorType = "kw";
  this.surfaceType = 2;
  this.isShowOutterDiamond = 0;
  this.isShowInnerDiamond = 0;
  this.innerDiamondColor = "dd";
  this.outterDiamondColor = "dd";
  this.shaderVertex = "";
  this.shaderFragment = "";
  this.bump_width = 0;
  this.bump_height = 0;
  this.bump_data = null;
  this.bump_dirty = false;
  this.bump_filename = "";
  this.env_filename = "";
  this.logo_filename = "";
  this.sand_filename = "";
  this.logoSc = 8.0; //8.0
  this.logoSize = 6.29; //6.29
  this.logoOffset = new THREE.Vector2(-0.1, 0.5);
  this.logoLeft = 0.0;
  this.initShader = function() {
    if (typeof RingShaderDict != "undefined") {
      scope.shaderVertex = RingShaderDict["vertexShader"].join("\n");
      scope.shaderFragment = RingShaderDict["fragmentShader"].join("\n");
    } else {
      $.ajax({
        url: "shaders/ringshader_vert.glsl",
        success: function(res) {
          scope.shaderVertex = res;
          if (scope.isShaderReady()) {
            scope.updateMesh();
          }
        },
        dataType: "text",
      });
      $.ajax({
        url: "shaders/ringshader_frag.glsl",
        success: function(res) {
          scope.shaderFragment = res;
          if (scope.isShaderReady()) {
            scope.updateMesh();
          }
        },
        dataType: "text",
      });
    }
  };
  this.initShader();
  this.isShaderReady = function() {
    if (this.shaderVertex.length > 0 && this.shaderFragment.length > 0) {
      return true;
    } else {
      return false;
    }
  };
  this.loadDiamond = function(callback) {
    var diamondPath = "models/diamond.stl";
    var path = "textures/bk/cube02_";
    var format = ".jpg";
    var urls = [
      path + "0" + format,
      path + "1" + format,
      path + "2" + format,
      path + "3" + format,
      path + "4" + format,
      path + "5" + format,
    ];
    var textureCube = THREE.ImageUtils.loadTextureCube(urls);
    textureCube.format = THREE.RGBFormat;
    textureCube.mapping = THREE.CubeReflectionMapping;
    var stl_loader = new THREE.STLLoader();
    stl_loader.load(diamondPath, function(geometry) {
      var shaderGenerator = new DiamondShaderGenerator();
      var material = shaderGenerator.generateShader(geometry);
      var center = geometry.boundingBox.getCenter();
      material.uniforms.iChannel0.value = textureCube;
      var g_diamond_mesh = new THREE.Mesh(geometry, material);
      var rotation = new THREE.Euler();
      rotation.set(scope.rotation.x, scope.rotation.y, scope.rotation.z + Math.PI * 0.75);
      var offset = new THREE.Vector3(0, 10.8, 0);
      offset.applyEuler(rotation);
      g_diamond_mesh.position.set(
        scope.position.x + offset.x,
        scope.position.y + offset.y,
        scope.position.z + offset.z
      );
      g_diamond_mesh.rotation.copy(rotation);
      var uMatrix = new THREE.Matrix4();
      uMatrix.makeRotationFromEuler(scope.rotation);
      g_diamond_mesh.material.uniforms["uMatrix"].value = uMatrix.getInverse(uMatrix);
      g_diamond_mesh.scale.set(0.3, 0.3, 0.3);
      g_diamond_mesh.visible = false;
      scope.diamond_mesh = g_diamond_mesh;
      material = shaderGenerator.generateShader(geometry);
      material.uniforms.iChannel0.value = textureCube;
      scope.inner_diamond_mesh = new THREE.Mesh(geometry.clone(), material);
      offset = new THREE.Vector3(0, -8, 0);
      var eular = new THREE.Euler();
      eular.set(scope.rotation.x, scope.rotation.y, scope.rotation.z - Math.PI * 0.25);
      offset.applyEuler(eular);
      scope.inner_diamond_mesh.position.set(
        scope.position.x + offset.x,
        scope.position.y + offset.y,
        scope.position.z + offset.z
      );
      scope.inner_diamond_mesh.rotation.set(scope.rotation.x, scope.rotation.y, scope.rotation.z + Math.PI * 1.25);
      scope.inner_diamond_mesh.scale.set(0.3, 0.3, 0.3);
      uMatrix = new THREE.Matrix4();
      uMatrix.makeRotationFromEuler(scope.inner_diamond_mesh.rotation);
      scope.inner_diamond_mesh.material.uniforms["uMatrix"].value = uMatrix.getInverse(uMatrix);
      scope.changeInnerDiamondColor(scope.innerDiamondColor);
      scope.changeOutterDiamondColor(scope.outterDiamondColor);
      scope.inner_diamond_mesh.visible = false;
      if (callback) {
        callback(scope.diamond_mesh, scope.inner_diamond_mesh);
      }
    });
  };
  this.loadEnvMap = function(filename) {
    if (this.env_filename == filename) {
      return;
    }
    this.env_filename = filename;
    var loader = new THREE.ImageLoader();
    var texture = scope.envMap;
    texture = THREE.ImageUtils.loadTexture(filename);
    texture.format = THREE.RGBAFormat;
    texture.magFilter = THREE.LinearFilter;
    texture.minFilter = THREE.LinearMipMapLinearFilter;
    texture.mapping = THREE.SphericalReflectionMapping;
    texture.wrapS = 1001;
    texture.wrapT = 1001;
    texture.anisotropy = true;
    scope.envMap = texture;
  };
  this.loadLogoMap = function(filename) {
    if (this.logo_filename == filename) {
      return;
    }
    this.logo_filename = filename;
    var texture = scope.logoMap;

    var loader = new THREE.ImageLoader();
    loader.load(filename, function(image) {
      scope.logoSize = image.width / image.height;
    });
    texture = THREE.ImageUtils.loadTexture(filename);
    texture.format = THREE.RGBFormat;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(10, 10);
    texture.anisotropy = true;
    scope.logoMap = texture;
  };
  this.updateGeometry = function() {
    this.makeRing();
    if (this.mesh) {
      this.mesh.geometry = this.geometry;
    }
  };
  this.loadSandMap = function(filename) {
    if (this.sand_filename == filename) {
      return;
    }
    this.sand_filename = filename;
    var loader = new THREE.ImageLoader();
    var texture = scope.sandMap;
    loader.load(filename, function(image) {
      texture.image = image;
      texture.wrapS = THREE.RepeatWrapping;
      texture.wrapT = THREE.RepeatWrapping;
      texture.needsUpdate = true;
    });
  };
  this.loadBumpMap = function(filename, repeat) {
    if (this.bump_filename == filename) {
      return;
    }
    this.bump_filename = filename;
    var loader = new THREE.ImageLoader();
    var texture = scope.bumpMap;
    if (!repeat) {
      repeat = [1, 1];
    }
    this.bump_dirty = false;
    scope.bumpMap.repeat.set(repeat[0], repeat[1]);
    loader.load(filename, function(image) {
      texture.image = image;
      texture.wrapS = THREE.RepeatWrapping;
      texture.wrapT = THREE.RepeatWrapping;
      texture.repeat.set(repeat[0], repeat[1]);
      texture.needsUpdate = true;
      if (scope.surfaceType == 3) {
      }
    });
  };
  this.showOutterDiamond = function(t) {
    this.isShowOutterDiamond = t;
    if (this.isShowOutterDiamond == 1) {
      var outter_offset = -0.45;
      if (this.widthType == "25") {
        outter_offset = -0.48;
        if (this.shapeType == "p") {
          outter_offset = -0.4;
        }
      } else {
        if (this.widthType == "40") {
          outter_offset = -0.45;
          if (this.shapeType == "p") {
            outter_offset = -0.4;
          }
        } else {
          if (this.widthType == "50") {
            outter_offset = -0.45;
            if (this.shapeType == "p") {
              outter_offset = -0.4;
            }
          } else {
            if (this.shapeType == "p") {
              outter_offset = -0.4;
            }
          }
        }
      }
      var rotation = new THREE.Euler();
      rotation.set(scope.rotation.x, scope.rotation.y, scope.rotation.z + Math.PI * 0.75);
      this.diamond_mesh.rotation.copy(rotation);
      var offset = new THREE.Vector3(0, this.radius + outter_offset, 0);
      offset.applyEuler(this.diamond_mesh.rotation);
      this.diamond_mesh.position.set(
        scope.position.x + offset.x,
        scope.position.y + offset.y,
        scope.position.z + offset.z
      );
      this.diamond_mesh.visible = true;
      this.updateMaterial();
    } else {
      this.diamond_mesh.visible = false;
      this.mesh.material.uniforms.uHoleRadius.value = -1;
    }
  };
  this.showInnerDiamond = function(t) {
    this.isShowInnerDiamond = t;
    var inner_offset = -1.32;
    if (this.widthType == "25") {
      inner_offset = -1.22;
    } else {
      if (this.widthType == "40") {
        inner_offset = -1.7;
        if (this.shapeType == "p") {
          inner_offset = -1.23;
        }
      } else {
        if (this.widthType == "50") {
          inner_offset = -1.7;
          if (this.shapeType == "p") {
            inner_offset = -1.63;
          }
        } else {
          if (this.shapeType == "p") {
            inner_offset = -1.2;
          }
        }
      }
    }
    if (this.isShowInnerDiamond == 1) {
      offset = new THREE.Vector3(0, -(this.radius + inner_offset), 0);
      scope.inner_diamond_mesh.rotation.set(scope.rotation.x, scope.rotation.y, scope.rotation.z + Math.PI * 1.25);
      offset.applyEuler(this.inner_diamond_mesh.rotation);
      this.inner_diamond_mesh.visible = true;
      this.inner_diamond_mesh.position.set(
        scope.position.x + offset.x,
        scope.position.y + offset.y,
        scope.position.z + offset.z
      );
      this.updateMaterial();
    } else {
      this.inner_diamond_mesh.visible = false;
      this.mesh.material.uniforms.uHoleRadius1.value = -1;
    }
  };
  this.makeRing = function() {
    if (!scope.base_ring) {
      return;
    }
    var rlen = scope.base_ring.length;
    var rsize = 90;
    if (this.surfaceType == 3) {
      rsize = 180;
    }
    var angle = (2 * Math.PI) / rsize;
    var max_px = 0;
    var nor_px = 0;
    var max_dy = 0;

    function getRadius() {
      var radius = 0;
      var max_uv = 0;
      for (var i = 1; i < scope.base_ring.length; i++) {
        var pos = scope.base_ring[i]["pos"];
        var uv_y = scope.base_ring[i]["uv"];
        if (uv_y > max_uv) {
          max_uv = uv_y;
        }
        if (Math.abs(pos[0]) > radius) {
          radius = Math.abs(pos[0]);
          max_px = pos[0];
          var nor = scope.base_ring[i]["nor"];
          nor_px = nor[0];
          var pos1 = scope.base_ring[i - 1]["pos"];
          var dy = pos[1] - pos1[1];
          max_dy = dy;
        }
      }
      console.log(max_px, max_dy, nor_px);
      if (max_px < 0) {
        for (var i = 0; i < scope.base_ring.length; i++) {
          var pos = scope.base_ring[i]["pos"];
          var nor = scope.base_ring[i]["nor"];
          scope.base_ring[i]["pos"] = [-pos[0], pos[1], pos[2]];
          scope.base_ring[i]["nor"] = [-nor[0], nor[1], nor[2]];
          if (scope.base_ring[i]["type"] == 1 && scope.shapeType == "p" && scope.widthType == "50") {
            scope.base_ring[i]["nor"] = [nor[0], nor[1], nor[2]];
          }
        }
        max_px = -max_px;
        nor_px = -nor_px;
      }
      if (max_dy < 0) {
        for (var i = 0; i < scope.base_ring.length / 2; i++) {
          var tmp = scope.base_ring[i];
          scope.base_ring[i] = scope.base_ring[scope.base_ring.length - 1 - i];
          scope.base_ring[scope.base_ring.length - 1 - i] = tmp;
        }
      }
      return radius;
    }
    var radius = getRadius();
    var dx = this.radius - radius;
    if (max_px < 0) {
      dx = -dx;
    }
    var base_ring = scope.base_ring;
    if (this.surfaceType == 3) {
      base_ring = [];
      var subNum = 5;
      for (var i = 0; i < scope.base_ring.length; i++) {
        base_ring.push(scope.base_ring[i]);
        if (scope.base_ring[i].type == 1) {
          var p0 = scope.base_ring[i]["pos"];
          var n0 = scope.base_ring[i]["nor"];
          var uv0 = scope.base_ring[i]["uv"];
          var p1 = scope.base_ring[(i + 1) % scope.base_ring.length]["pos"];
          var n1 = scope.base_ring[(i + 1) % scope.base_ring.length]["nor"];
          var uv1 = scope.base_ring[(i + 1) % scope.base_ring.length]["uv"];
          for (var j = 1; j < subNum; j++) {
            var p2 = [0, 0, 0];
            var n2 = [0, 0, 0];
            var uv2 = 0;
            var a = (subNum - j) / (subNum + 1e-9);
            for (var k = 0; k < 3; k++) {
              p2[k] = p0[k] * a + p1[k] * (1 - a);
              n2[k] = n0[k] * a + n1[k] * (1 - a);
            }
            uv2 = uv0 * a + uv1 * (1 - a);
            base_ring.push({
              pos: p2,
              nor: n2,
              uv: uv2,
              type: 1,
            });
          }
        }
      }
      console.log("len", base_ring.length);
    }
    rlen = base_ring.length;
    radius = this.radius;
    var tlen = 2 * Math.PI * radius;
    var tval = 0;

    function getBumpValue(uv) {
      if (!scope.bump_data) {
        return 0;
      }
      var tuv = uv.clone();
      tuv = tuv.applyMatrix3(scope.bumpMap.matrix);
      var x = tuv.x - Math.floor(tuv.x);
      var y = tuv.y - Math.floor(tuv.y);
      x = Math.floor(scope.bump_width * x);
      y = Math.floor(scope.bump_height * y);
      return scope.bump_data[4 * (y * scope.bump_width + x) + 1] - 255;
    }

    function getVertex(i, j) {
      var pos = base_ring[j]["pos"];
      var nor = base_ring[j]["nor"];
      var uv_y = base_ring[j]["uv"];
      pos = new THREE.Vector3(pos[0] + dx, pos[2], pos[1]);
      nor = new THREE.Vector3(nor[0], nor[2], nor[1]);
      var quat = new THREE.Quaternion();
      if (max_dy > 0) {
        quat.setFromAxisAngle(new THREE.Vector3(0, 0, 1), angle * i);
      } else {
        quat.setFromAxisAngle(new THREE.Vector3(0, 0, 1), angle * i);
      }
      pos = pos.applyQuaternion(quat);
      nor = nor.applyQuaternion(quat);
      var uv;
      if (base_ring[j]["type"] == 1 || (scope.surfaceType == 3 && base_ring[j]["type"] % 2 == 1)) {
        uv_y = uv_y / tlen;
        var uv_x = i / (rsize + 1e-8);
        uv = new THREE.Vector2(uv_x, uv_y);
      } else {
        if (base_ring[j]["type"] == 2 && scope.isShaderReady()) {
          uv_y = uv_y / tlen;
          var uv_x = -(angle * i) / (2 * Math.PI);
          uv = new THREE.Vector2(uv_x, uv_y);
        } else {
          uv = new THREE.Vector2(0, 0);
        }
      }
      return [pos, nor, uv];
    }
    var faceNum = rsize * rlen * 2;
    var vertices = new Float32Array(faceNum * 3 * 3);
    var normals = new Float32Array(faceNum * 3 * 3);
    var uvs = new Float32Array(faceNum * 3 * 2);
    for (var i = 0; i < rsize; i++) {
      for (var j = 0; j < rlen; j++) {
        var v0 = getVertex(i, j);
        var v1 = getVertex(i + 1, j);
        var v2 = getVertex(i + 1, (j + 1) % rlen);
        var v3 = getVertex(i, (j + 1) % rlen);
        if (v0[2].x == 0 || v1[2].x == 0 || v2[2].x == 0 || v3[2].x == 0) {
          v0[2].set(0, 0);
          v1[2].set(0, 0);
          v2[2].set(0, 0);
          v3[2].set(0, 0);
        }
        if (v0[2].x < 0 || v1[2].x < 0 || v2[2].x < 0 || v3[2].x < 0) {
          if (v0[2].x > 0) {
            v0[2].x = -v0[2].x;
          }
          if (v1[2].x > 0) {
            v1[2].x = -v1[2].x;
          }
          if (v2[2].x > 0) {
            v2[2].x = -v2[2].x;
          }
          if (v3[2].x > 0) {
            v3[2].x = -v3[2].x;
          }
        }
        var fid = 2 * (i * rlen + j);
        vertices[3 * 3 * fid + 3 * 0 + 0] = v0[0].x;
        vertices[3 * 3 * fid + 3 * 0 + 1] = v0[0].y;
        vertices[3 * 3 * fid + 3 * 0 + 2] = v0[0].z;
        vertices[3 * 3 * fid + 3 * 1 + 0] = v1[0].x;
        vertices[3 * 3 * fid + 3 * 1 + 1] = v1[0].y;
        vertices[3 * 3 * fid + 3 * 1 + 2] = v1[0].z;
        vertices[3 * 3 * fid + 3 * 2 + 0] = v2[0].x;
        vertices[3 * 3 * fid + 3 * 2 + 1] = v2[0].y;
        vertices[3 * 3 * fid + 3 * 2 + 2] = v2[0].z;
        normals[3 * 3 * fid + 3 * 0 + 0] = v0[1].x;
        normals[3 * 3 * fid + 3 * 0 + 1] = v0[1].y;
        normals[3 * 3 * fid + 3 * 0 + 2] = v0[1].z;
        normals[3 * 3 * fid + 3 * 1 + 0] = v1[1].x;
        normals[3 * 3 * fid + 3 * 1 + 1] = v1[1].y;
        normals[3 * 3 * fid + 3 * 1 + 2] = v1[1].z;
        normals[3 * 3 * fid + 3 * 2 + 0] = v2[1].x;
        normals[3 * 3 * fid + 3 * 2 + 1] = v2[1].y;
        normals[3 * 3 * fid + 3 * 2 + 2] = v2[1].z;
        uvs[2 * 3 * fid + 2 * 0 + 0] = v0[2].x;
        uvs[2 * 3 * fid + 2 * 0 + 1] = v0[2].y;
        uvs[2 * 3 * fid + 2 * 1 + 0] = v1[2].x;
        uvs[2 * 3 * fid + 2 * 1 + 1] = v1[2].y;
        uvs[2 * 3 * fid + 2 * 2 + 0] = v2[2].x;
        uvs[2 * 3 * fid + 2 * 2 + 1] = v2[2].y;
        fid = 2 * (i * rlen + j) + 1;
        vertices[3 * 3 * fid + 3 * 0 + 0] = v2[0].x;
        vertices[3 * 3 * fid + 3 * 0 + 1] = v2[0].y;
        vertices[3 * 3 * fid + 3 * 0 + 2] = v2[0].z;
        vertices[3 * 3 * fid + 3 * 1 + 0] = v3[0].x;
        vertices[3 * 3 * fid + 3 * 1 + 1] = v3[0].y;
        vertices[3 * 3 * fid + 3 * 1 + 2] = v3[0].z;
        vertices[3 * 3 * fid + 3 * 2 + 0] = v0[0].x;
        vertices[3 * 3 * fid + 3 * 2 + 1] = v0[0].y;
        vertices[3 * 3 * fid + 3 * 2 + 2] = v0[0].z;
        normals[3 * 3 * fid + 3 * 0 + 0] = v2[1].x;
        normals[3 * 3 * fid + 3 * 0 + 1] = v2[1].y;
        normals[3 * 3 * fid + 3 * 0 + 2] = v2[1].z;
        normals[3 * 3 * fid + 3 * 1 + 0] = v3[1].x;
        normals[3 * 3 * fid + 3 * 1 + 1] = v3[1].y;
        normals[3 * 3 * fid + 3 * 1 + 2] = v3[1].z;
        normals[3 * 3 * fid + 3 * 2 + 0] = v0[1].x;
        normals[3 * 3 * fid + 3 * 2 + 1] = v0[1].y;
        normals[3 * 3 * fid + 3 * 2 + 2] = v0[1].z;
        uvs[2 * 3 * fid + 2 * 0 + 0] = v2[2].x;
        uvs[2 * 3 * fid + 2 * 0 + 1] = v2[2].y;
        uvs[2 * 3 * fid + 2 * 1 + 0] = v3[2].x;
        uvs[2 * 3 * fid + 2 * 1 + 1] = v3[2].y;
        uvs[2 * 3 * fid + 2 * 2 + 0] = v0[2].x;
        uvs[2 * 3 * fid + 2 * 2 + 1] = v0[2].y;
      }
    }
    base_ring = [];
    if (!this.geometry) {
      this.geometry = new THREE.BufferGeometry();
      this.geometry.addAttribute("position", new THREE.BufferAttribute(vertices, 3));
      this.geometry.addAttribute("normal", new THREE.BufferAttribute(normals, 3));
      this.geometry.addAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    } else {
      this.geometry.removeAttribute("position");
      this.geometry.removeAttribute("normal");
      this.geometry.removeAttribute("uv");
      this.geometry.addAttribute("position", new THREE.BufferAttribute(vertices, 3));
      this.geometry.addAttribute("normal", new THREE.BufferAttribute(normals, 3));
      this.geometry.addAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    }
    return radius;
  };
  this.updateMesh = function() {
    this.updateMaterial();
    var radius = this.makeRing();
    this.position.y = radius - 10;
    this.position.x = 10 - radius;
    if (!this.mesh) {
      this.mesh = new THREE.Mesh(this.geometry, this.material);
      this.mesh.position.copy(this.position);
      this.mesh.rotation.copy(this.rotation);
    } else {
      this.mesh.position.copy(this.position);
      this.mesh.rotation.copy(this.rotation);
    }
    var material = this.material;
    if (this.isShowMirror && !this.mirror_mesh) {
      this.mirror_mesh = new THREE.Mesh(this.mesh.geometry, material);
      this.mirror_mesh.rotation.set(this.mesh.rotation.x + Math.PI, -this.mesh.rotation.y, this.mesh.rotation.z);
      this.mirror_mesh.position.set(this.mesh.position.x, this.mesh.position.y - radius * 2, this.mesh.position.z);
    } else {
    }
  };
  this.load = function(file_url, callback) {
    if (typeof RingModelDict != "undefined") {
      scope.base_ring = RingModelDict[scope.shapeType + scope.widthType];
      scope.updateMesh();
      if (callback) {
        callback(scope.mesh, scope.mirror_mesh);
      }
    } else {
      $.ajax({
        url: file_url,
        success: function(res) {
          scope.base_ring = res;
          scope.updateMesh();
          if (callback) {
            callback(scope.mesh, scope.mirror_mesh);
          }
        },
        dataType: "json",
      });
    }
  };
  this.updateColorMaps = function(callback) {
    var env_filename = "textures/";
    var logo_filename = "";
    if (this.colorType == "pt") {
      env_filename += "ring5bw.png";
      logo_filename = "textures/logo_spaced.jpg";
    } else {
      if (this.colorType == "kw") {
        env_filename += "ring5bw.png";
        logo_filename = "textures/logo2_spaced.jpg";
      } else {
        if (this.colorType == "kr") {
          env_filename += "18kr.png";
          logo_filename = "textures/logo2_spaced.jpg";
        } else {
          env_filename += "18ky.png";
          logo_filename = "textures/logo2_spaced.jpg";
        }
      }
    }
    if (this.env_filename != env_filename) {
      this.loadLogoMap(logo_filename);
      this.env_filename = env_filename;
      var loader = new THREE.ImageLoader();
      loader.load(env_filename, function(image) {
        scope.envMap.image = image;
        scope.envMap.needsUpdate = true;
        if (callback) {
          callback();
        }
      });
    } else {
      if (this.logo_filename != logo_filename) {
        var loader = new THREE.ImageLoader();
        loader.load(logo_filename, function(image) {
          scope.logoMap.image = image;
          scope.logoMap.needsUpdate = true;
          if (callback) {
            callback();
          }
        });
      }
    }
  };
  this.updateMaterial = function() {
    var env_filename = "textures/";
    var diffuse = new THREE.Vector3(1.455, 1.4666, 1.47);
    var bumpscale = 0;
    var metalness = 0.93;
    var displaceArea = 0.023;
    if (this.widthType == "25") {
      displaceArea = 0.016;
    } else {
      if (this.widthType == "40") {
        displaceArea = 0.027;
      } else {
        if (this.widthType == "50") {
          displaceArea = 0.035;
        }
      }
    }
    if (this.colorType == "pt") {
      env_filename += "ring5bw.png";
      this.loadLogoMap("textures/logo_spaced.jpg"); //logo3
    } else {
      if (this.colorType == "kw") {
        env_filename += "ring5bw.png";
        this.loadLogoMap("textures/logo2_spaced.jpg");
      } else {
        if (this.colorType == "kr") {
          env_filename += "18kr.png";
          diffuse = new THREE.Vector3(1.454, 1.4666, 1.47);
          this.loadLogoMap("textures/logo2_spaced.jpg");
          metalness = 0.8;
          bumpscale = 0.06;
        } else {
          if (this.colorType == "ky") {
            env_filename += "18ky.png";
            diffuse = new THREE.Vector3(1.355, 1.3266, 1.2);
            this.loadLogoMap("textures/logo2_spaced.jpg");
            metalness = 0.92;
            bumpscale = 0.15;
          }
        }
      }
    }
    this.loadEnvMap(env_filename);
    if (this.surfaceType == 0) {
      bumpscale = 0;
    } else {
      if (this.surfaceType == 1) {
        this.loadBumpMap("textures/hammer.jpg");
        bumpscale = 0.12;
      } else {
        if (this.surfaceType == 2) {
          this.loadBumpMap("textures/bump0.jpg");
          bumpscale = 0.015;
        } else {
          if (this.surfaceType == 3) {
            this.loadBumpMap("textures/linear.jpg");
            if (metalness > 0.9) {
              metalness = 0.9;
            }
            bumpscale = 0.8;
          } else {
            if (this.surfaceType == 4) {
              this.loadBumpMap("textures/sand.jpg");
              bumpscale = 0.1;
            }
          }
        }
      }
    }
    this.loadSandMap("textures/sand.jpg");
    var roughness = 1 - metalness;
    var offset = scope.bumpMap.offset;
    var repeat = scope.bumpMap.repeat;
    var rotation = scope.bumpMap.rotation;
    var center = scope.bumpMap.center;
    if (scope.surfaceType == 1) {
      repeat.set(11, 14);
    } else {
      if (scope.surfaceType == 2) {
        repeat.set(12, 12);
        rotation = -Math.PI / 4;
      } else {
        if (scope.surfaceType == 3) {
          repeat.set(1, 14);
        } else {
          if (scope.surfaceType == 4) {
            repeat.set(10, 10);
          }
        }
      }
    }
    scope.bumpMap.matrix.setUvTransform(offset.x, offset.y, repeat.x, repeat.y, rotation, center.x, center.y);
    if (this.isShaderReady()) {
      var holeRadius = -1;
      if (this.isShowOutterDiamond == 1) {
        holeRadius = 0.012;
      }
      var holeRadius1 = -1;
      if (this.isShowInnerDiamond == 1) {
        holeRadius1 = 0.014;
      }
      if (this.material && typeof this.material.uniforms != "undefined") {
        console.log(scope.logoSc, scope.logoOffset);
        this.material.uniforms.envMap.value = scope.envMap;
        this.material.uniforms.bumpMap.value = scope.bumpMap;
        this.material.uniforms.bumpScale.value = bumpscale;
        this.material.uniforms.diffuse.value.copy(diffuse);
        this.material.uniforms.metalness.value = metalness;
        this.material.uniforms.roughness.value = roughness;
        this.material.uniforms.uSurfaceType.value = scope.surfaceType;
        this.material.uniforms.uvTransform.value = scope.bumpMap.matrix;
        this.material.uniforms.uHoleRadius.value = holeRadius;
        this.material.uniforms.uHoleRadius1.value = holeRadius1;
        this.material.uniforms.logoMap.value = scope.logoMap;
        this.material.uniforms.displaceArea.value = displaceArea;
        this.material.uniforms.logoSc.value = scope.logoSc;
        this.material.uniforms.logoLeft.value = scope.logoLeft;
        this.material.uniforms.logoSize.value = scope.logoSize;
        this.material.uniforms.logoOffset.value = scope.logoOffset;
      } else {
        this.material = new THREE.ShaderMaterial({
          uniforms: {
            envMap: {
              value: scope.envMap,
            },
            bumpMap: {
              value: scope.bumpMap,
            },
            bumpScale: {
              value: bumpscale,
            },
            diffuse: {
              value: diffuse,
            },
            emissive: {
              value: new THREE.Vector3(0, 0, 0),
            },
            metalness: {
              value: metalness,
            },
            roughness: {
              value: roughness,
            },
            uSurfaceType: {
              value: scope.surfaceType,
            },
            opacity: {
              value: 1,
            },
            flipEnvMap: {
              value: 1,
            },
            maxMipLevel: {
              value: 3,
            },
            refractionRatio: {
              value: 0.98,
            },
            reflectivity: {
              value: 0.5,
            },
            envMapIntensity: {
              value: 1,
            },
            uvTransform: {
              value: scope.bumpMap.matrix,
            },
            uHoleCenter: {
              value: new THREE.Vector2(0.625, 0),
            },
            uHoleRadius: {
              value: holeRadius,
            },
            logoSc: {
              value: scope.logoSc,
            },
            logoSize: {
              value: scope.logoSize,
            },
            logoOffset: {
              value: scope.logoOffset,
            },
            logoLeft: {
              value: scope.logoLeft,
            },
            uHoleCenter1: {
              value: new THREE.Vector2(-0.375, 0),
            },
            uHoleRadius1: {
              value: holeRadius1,
            },
            logoMap: {
              value: scope.logoMap,
            },
            displaceArea: {
              value: displaceArea,
            },
            sandMap: {
              value: scope.sandMap,
            },
          },
          vertexShader: scope.shaderVertex,
          fragmentShader: scope.shaderFragment,
          transparent: true,
          flatShading: false,
          side: THREE.DoubleSide,
          extensions: {
            derivatives: true,
          },
        });
      }
    } else {
      this.material = new THREE.MeshStandardMaterial({
        map: null,
        bumpMap: scope.bumpMap,
        bumpScale: bumpscale,
        normalMap: null,
        color: scope.diffuseColor,
        metalness: metalness,
        roughness: roughness,
        flatShading: false,
        envMap: scope.envMap,
      });
    }
    if (this.mesh) {
      this.mesh.material = this.material;
    }
    if (this.isShowMirror && this.mirror_mesh) {
      var matrix = new THREE.Matrix3();
      matrix.setUvTransform(offset.x, offset.y, repeat.x, -repeat.y, rotation, center.x, center.y);
      this.mirror_mesh.material = this.material.clone();
      this.mirror_mesh.material.uniforms.uvTransform.value = matrix;
      this.mirror_mesh.material.uniforms.envMap.value = scope.envMap;
      this.mirror_mesh.material.uniforms.bumpMap.value = scope.bumpMap;
      this.mirror_mesh.material.uniforms.logoMap.value = scope.logoMap;
      this.mirror_mesh.material.uniforms.uHoleRadius.value = -1;
      this.mirror_mesh.material.uniforms.uHoleRadius1.value = -1;
    }
  };
  this.changeBumpScale = function(sc) {
    this.mesh.material.uniforms.bumpScale.value = sc;
  };
  this.updateModel = function(callback) {
    var file_url = "models/" + this.shapeType + this.widthType + "a.json";
    this.load(file_url, callback);
  };
  this.changeShapeType = function(type) {
    this.shapeType = type;
    this.updateModel();
    this.showInnerDiamond(this.isShowInnerDiamond);
    this.showOutterDiamond(this.isShowOutterDiamond);
  };
  this.changeWidthType = function(type) {
    this.widthType = type;
    this.updateModel();
    this.showInnerDiamond(this.isShowInnerDiamond);
    this.showOutterDiamond(this.isShowOutterDiamond);
  };
  this.changeColorType = function(type) {
    this.colorType = type;
    this.updateColorMaps(function() {
      scope.updateMaterial();
    });
  };
  this.changeSurfaceType = function(type) {
    this.surfaceType = type;
    this.updateMesh();
  };
  this.updateDiamondMaterial = function(mesh, pow, color, intensity) {
    var vcolor = new THREE.Vector3();
    if (color.length >= 3) {
      vcolor.set(color[0], color[1], color[2]);
    } else {
      vcolor.set(color, color, color);
    }
    var material = mesh.material;
    material.uniforms.iObjColor.value.copy(vcolor);
    if (intensity) {
      material.uniforms.uIntensity.value = intensity;
    } else {
      material.uniforms.uIntensity.value = 0.5;
    }
    material.uniforms.uPow.value = pow;
  };
  this.changeInnerDiamondColor = function(type) {
    this.innerDiamondColor = type;
    var res = DiamondColorDict[type];
    this.updateDiamondMaterial(this.inner_diamond_mesh, res[1], res[0], res[2]);
  };
  this.changeOutterDiamondColor = function(type) {
    this.outterDiamondColor = type;
    var res = DiamondColorDict[type];
    this.updateDiamondMaterial(this.diamond_mesh, res[1], res[0], res[2]);
  };
  this.show = function(t) {
    if (t) {
      this.mesh.visible = true;
    } else {
      this.mesh.visible = false;
      this.inner_diamond_mesh.visible = false;
      this.diamond_mesh.visible = false;
    }
  };
};
