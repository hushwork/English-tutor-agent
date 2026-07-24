// Camera Tutor — Live2D Pipe Renderer
//
// Minimal C++ program built on Cubism Native SDK that:
//   1. Loads a .moc3 model
//   2. Reads viseme parameters from stdin (binary protocol)
//   3. Renders one frame and outputs RGBA pixels to stdout
//
// Protocol (binary, pipe):
//   stdin ←  [uint32 param_count][uint32 name_len][char* name][float value]...
//   stdout → [uint32 frame_size][uint8* rgba_pixels]
//
// Build: see scripts/build_live2d_linux.sh
//
// Dependencies: Cubism Native SDK (https://github.com/Live2D/CubismNativeSamples)
//
// This file replaces the standard Demo/main.cpp in the SDK sample.
// Copy it to CubismNativeSamples/Samples/OpenGL/Demo/proj.linux.cmake/src/
// and rebuild.

#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <GL/glew.h>
#include <GL/gl.h>

// Cubism SDK headers (from SDK include path)
#include <CubismFramework.hpp>
#include <CubismModelSettingJson.hpp>
#include <CubismUserModel.hpp>
#include <ICubismModelSetting.hpp>
#include <Rendering/OpenGL/CubismRenderer_OpenGLES2.hpp>
#include <Motion/CubismMotion.hpp>

using namespace Live2D::Cubism::Framework;
using namespace Live2D::Cubism::Framework::Rendering;

// ── Minimal Cubism User Model ────────────────────────────────────

class PipeModel : public CubismUserModel {
public:
    bool LoadModel(const char* modelDir, const char* modelJsonName) {
        // Load model settings
        std::string jsonPath = std::string(modelDir) + "/" + modelJsonName;
        Csm::ICubismModelSetting* setting = new CubismModelSettingJson(jsonPath.c_str());

        // Load .moc3
        std::string mocPath = std::string(modelDir) + "/" + setting->GetModelFileName();
        LoadModel(mocPath.c_str());

        // Create renderer
        CreateRenderer();

        // Initialize renderer
        GetRenderer()->Initialize(Csm::CubismMatrix44());

        delete setting;
        return true;
    }

private:
    // Minimal: no physics, no pose, no eye blink — just the face mesh
};

// ── Global state ─────────────────────────────────────────────────

static PipeModel* g_model = nullptr;
static int g_width = 512;
static int g_height = 512;
static GLuint g_fbo = 0;
static GLuint g_fbo_texture = 0;
static GLuint g_fbo_depth = 0;

// Parameter cache
static std::unordered_map<std::string, Csm::csmFloat32*> g_paramCache;

// ── Offscreen rendering setup ────────────────────────────────────

void InitOffscreen(int width, int height) {
    g_width = width;
    g_height = height;

    // Framebuffer
    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);

    // Color texture
    glGenTextures(1, &g_fbo_texture);
    glBindTexture(GL_TEXTURE_2D, g_fbo_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, g_fbo_texture, 0);

    // Depth buffer
    glGenRenderbuffers(1, &g_fbo_depth);
    glBindRenderbuffer(GL_RENDERBUFFER, g_fbo_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, g_fbo_depth);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ── Frame output ─────────────────────────────────────────────────

void OutputFrame() {
    std::vector<uint8_t> pixels(g_width * g_height * 4);

    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glReadPixels(0, 0, g_width, g_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Write to stdout: [uint32 size] [pixel data]
    uint32_t size = static_cast<uint32_t>(pixels.size());
    fwrite(&size, sizeof(uint32_t), 1, stdout);
    fwrite(pixels.data(), 1, pixels.size(), stdout);
    fflush(stdout);
}

// ── Parameter input ──────────────────────────────────────────────

void CacheParameters() {
    // Build parameter name → pointer cache for fast lookup
    Csm::csmInt32 paramCount = g_model->GetModel()->GetParameterCount();
    for (int i = 0; i < paramCount; i++) {
        const Csm::csmChar* id = g_model->GetModel()->GetParameterId(i);
        Csm::csmFloat32* ptr = g_model->GetModel()->GetParameterValuePtr(i);
        g_paramCache[std::string(id)] = ptr;
    }
}

void ReadAndApplyParameters() {
    // Read: [uint32 count] then for each: [uint32 name_len] [char* name] [float value]
    uint32_t count;
    if (fread(&count, sizeof(uint32_t), 1, stdin) != 1) return;

    for (uint32_t i = 0; i < count; i++) {
        // Read name
        uint32_t nameLen;
        if (fread(&nameLen, sizeof(uint32_t), 1, stdin) != 1) return;

        std::vector<char> nameBuf(nameLen + 1, 0);
        if (fread(nameBuf.data(), 1, nameLen, stdin) != nameLen) return;

        // Read value
        float value;
        if (fread(&value, sizeof(float), 1, stdin) != 1) return;

        // Apply
        std::string name(nameBuf.data(), nameLen);
        auto it = g_paramCache.find(name);
        if (it != g_paramCache.end()) {
            *(it->second) = value;
        }
    }
}

// ── Main ─────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // Parse args
    const char* modelDir = "models/Haru";
    const char* modelJson = "Haru.model3.json";
    int width = 512;
    int height = 512;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            modelDir = argv[++i];
            // Extract directory and json filename
            // Model path format: "models/Haru/Haru.model3.json"
            // For simplicity, assume modelDir is the directory path
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = atoi(argv[++i]);
        }
    }

    // Create minimal GL context (headless/offscreen)
    // On Orin with EGL, create a Pbuffer surface
    // For simplicity, assume a display is available
    // In production, use EGL_PBUFFER_BIT for true headless

    // Init Cubism
    CubismFramework::Option option;
    CubismFramework::StartUp(&option);
    CubismFramework::Initialize();

    // Load model
    g_model = new PipeModel();
    g_model->LoadModel(modelDir, modelJson);
    CacheParameters();

    // Init offscreen rendering
    InitOffscreen(width, height);

    // Main loop: read params → update → render → output
    while (true) {
        ReadAndApplyParameters();

        // Update model
        g_model->GetModel()->Update();
        g_model->GetModel()->Update();
        g_model->Update(0.016f); // ~60fps delta

        // Render
        glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
        glViewport(0, 0, width, height);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        Csm::CubismMatrix44 projection;
        projection.Scale(1.0f, static_cast<float>(width) / static_cast<float>(height));
        g_model->GetRenderer()->SetMvpMatrix(&projection);
        g_model->GetRenderer()->DrawModel();

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Output frame
        OutputFrame();
    }

    // Cleanup (never reached in pipe mode)
    delete g_model;
    CubismFramework::Dispose();
    return 0;
}
