// SPDX-License-Identifier: GPL-3.0-or-later
// The host processor: what it is and how busy the game keeps it (nxrt/cpu_info.h). For the F1 menu's Debug page and one log line at start.
#include <windows.h>
#include <psapi.h>
#include <intrin.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "nxrt/cpu_info.h"
#include "nxrt/diag.h"

namespace nxrt::cpu {
namespace {

uint64_t ft64(const FILETIME& f) { return (static_cast<uint64_t>(f.dwHighDateTime) << 32) | f.dwLowDateTime; }
uint64_t now_100ns() {
    FILETIME f;
    GetSystemTimeAsFileTime(&f);
    return ft64(f);
}

int bits(uint64_t m) {
    int n = 0;
    for (; m; m &= m - 1) ++n;
    return n;
}

void read_cpuid(Static& s) {
    int b[4] = {};
    __cpuid(b, 0);
    const int max_leaf = b[0];
    char vendor[13] = {};
    std::memcpy(vendor, &b[1], 4);
    std::memcpy(vendor + 4, &b[3], 4);
    std::memcpy(vendor + 8, &b[2], 4);
    s.vendor = vendor;
    __cpuid(b, static_cast<int>(0x80000000u));
    if (static_cast<unsigned>(b[0]) >= 0x80000004u) {
        char brand[49] = {};
        for (unsigned i = 0; i < 3; i++) __cpuid(reinterpret_cast<int*>(brand) + 4 * i, static_cast<int>(0x80000002u + i));
        std::string t = brand;
        const size_t a = t.find_first_not_of(' ');
        const size_t z = t.find_last_not_of(' ');
        s.brand = a == std::string::npos ? std::string() : t.substr(a, z - a + 1);
    }
    unsigned xcr0 = 0;
    if (max_leaf >= 1) {
        __cpuid(b, 1);
        const unsigned ecx = static_cast<unsigned>(b[2]);
        s.sse42 = (ecx >> 20) & 1;
        s.popcnt = (ecx >> 23) & 1;
        s.cx16 = (ecx >> 13) & 1;
        if ((ecx >> 27) & 1) {   // OSXSAVE: the OS says which register state it saves
            unsigned hi = 0;
            __asm__ volatile("xgetbv" : "=a"(xcr0), "=d"(hi) : "c"(0));
        }
        const bool ymm = (xcr0 & 6) == 6;
        s.avx = ymm && ((ecx >> 28) & 1);
        s.fma = ymm && ((ecx >> 12) & 1);
    }
    if (max_leaf >= 7) {
        __cpuidex(b, 7, 0);
        const unsigned ebx = static_cast<unsigned>(b[1]);
        s.avx2 = s.avx && ((ebx >> 5) & 1);
        s.bmi2 = (ebx >> 8) & 1;
        s.avx512f = (xcr0 & 0xe6) == 0xe6 && ((ebx >> 16) & 1);   // and the OS saves the opmask and ZMM state
    }
}

void read_topology(Static& s) {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
    if (!len) return;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationAll, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) return;
    struct Core {
        int threads;
        int cls;
    };
    std::vector<Core> cores;
    int lo = 255, hi = 0;
    for (DWORD off = 0; off < len;) {
        const auto* e = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        if (e->Relationship == RelationProcessorCore) {
            int t = 0;
            for (WORD g = 0; g < e->Processor.GroupCount; ++g) t += bits(static_cast<uint64_t>(e->Processor.GroupMask[g].Mask));
            const int cls = e->Processor.EfficiencyClass;
            cores.push_back({t, cls});
            lo = std::min(lo, cls);
            hi = std::max(hi, cls);
        } else if (e->Relationship == RelationCache) {
            if (e->Cache.Level == 2) s.l2_kb += static_cast<int>(e->Cache.CacheSize / 1024);
            if (e->Cache.Level == 3) s.l3_kb += static_cast<int>(e->Cache.CacheSize / 1024);
        }
        off += e->Size;
    }
    s.cores = static_cast<int>(cores.size());
    for (const Core& c : cores) {
        s.logical += c.threads;
        if (hi > lo) {   // hybrid: the highest class is the performance cores
            if (c.cls == hi) s.perf_cores++, s.perf_threads += c.threads;
            else s.eff_cores++, s.eff_threads += c.threads;
        }
    }
}

Static read_static() {
    Static s;
    read_cpuid(s);
    read_topology(s);
    if (s.logical <= 0) {
        SYSTEM_INFO si{};
        GetSystemInfo(&si);
        s.logical = static_cast<int>(si.dwNumberOfProcessors);
    }
    DWORD mhz = 0, size = sizeof mhz;
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", L"~MHz", RRF_RT_REG_DWORD, nullptr, &mhz, &size) ==
        ERROR_SUCCESS)
        s.base_mhz = static_cast<int>(mhz);
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) s.ram_total = ms.ullTotalPhys;
    return s;
}

std::mutex g_mu;
uint64_t g_wall = 0, g_proc = 0, g_sys_idle = 0, g_sys_busy = 0;   // the previous sample (100 ns units)
uint64_t g_thread_wall = 0;
std::unordered_map<uint32_t, uint64_t> g_thread_prev;              // host tid -> kernel + user time at the previous call
std::atomic<const char*> g_game_build{nullptr};

}  // namespace

void set_game_build(const char* text) { g_game_build.store(text); }
const char* game_build() { return g_game_build.load(); }

const Static& info() {
    static const Static s = read_static();
    return s;
}

std::string features_text(const Static& s) {
    std::string t;
    auto add = [&](bool on, const char* name) {
        if (!on) return;
        if (!t.empty()) t += ", ";
        t += name;
    };
    add(s.sse42, "SSE4.2");
    add(s.popcnt, "POPCNT");
    add(s.avx, "AVX");
    add(s.avx2, "AVX2");
    add(s.fma, "FMA3");
    add(s.bmi2, "BMI2");
    add(s.avx512f, "AVX-512");
    return t.empty() ? std::string("baseline x86-64") : t;
}

std::string summary_line() {
    const Static& s = info();
    char buf[512];
    char split[64] = "";
    if (s.perf_cores || s.eff_cores) std::snprintf(split, sizeof split, " (%d performance, %d efficiency)", s.perf_cores, s.eff_cores);
    std::snprintf(buf, sizeof buf, "%s [%s]: %d cores%s, %d threads, L2 %.1f MB, L3 %.1f MB, base %.2f GHz, %s; RAM %.1f GB",
                  s.brand.empty() ? "unknown processor" : s.brand.c_str(), s.vendor.c_str(), s.cores, split, s.logical, s.l2_kb / 1024.0,
                  s.l3_kb / 1024.0, s.base_mhz / 1000.0, features_text(s).c_str(), static_cast<double>(s.ram_total) / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

// The Windows power plan and power mode (FR7, as Fritchii logs them): "Power saver", a battery saver mode or a battery that is not charging
// lowers the processor's clocks, a common cause of low frame rates in player logs. powrprof.dll is loaded here, not linked.
std::string power_line() {
    std::string out;
    HMODULE pp = LoadLibraryW(L"powrprof.dll");
    if (pp) {
        using GetActive = DWORD(WINAPI*)(HKEY, GUID**);
        using ReadName = DWORD(WINAPI*)(HKEY, const GUID*, const GUID*, const GUID*, UCHAR*, DWORD*);
        using GetOverlay = DWORD(WINAPI*)(GUID*);
        auto get_active = reinterpret_cast<GetActive>(GetProcAddress(pp, "PowerGetActiveScheme"));
        auto read_name = reinterpret_cast<ReadName>(GetProcAddress(pp, "PowerReadFriendlyName"));
        auto get_overlay = reinterpret_cast<GetOverlay>(GetProcAddress(pp, "PowerGetEffectiveOverlayScheme"));   // Windows 10 1809 and newer
        GUID* scheme = nullptr;
        if (get_active && get_active(nullptr, &scheme) == ERROR_SUCCESS && scheme) {
            static const struct { GUID g; const char* name; } kKnown[] = {
                {{0x381b4222, 0xf694, 0x41f0, {0x96, 0x85, 0xff, 0x5b, 0xb2, 0x60, 0xdf, 0x2e}}, "Balanced"},
                {{0x8c5e7fda, 0xe8bf, 0x4a96, {0x9a, 0x85, 0xa6, 0xe2, 0x3a, 0x8c, 0x63, 0x5c}}, "High performance"},
                {{0xa1841308, 0x3541, 0x4fab, {0xbc, 0x81, 0xf7, 0x15, 0x56, 0xf2, 0x0b, 0x4a}}, "Power saver"},
                {{0xe9a42b02, 0xd5df, 0x448d, {0xaa, 0x00, 0x03, 0xf1, 0x47, 0x49, 0xeb, 0x61}}, "Ultimate Performance"},
            };
            const char* known = nullptr;
            for (const auto& k : kKnown)
                if (IsEqualGUID(*scheme, k.g)) known = k.name;
            std::string name = known ? known : "";
            wchar_t wname[128] = {};
            DWORD sz = sizeof wname;
            if (!known && read_name && read_name(nullptr, scheme, nullptr, nullptr, reinterpret_cast<UCHAR*>(wname), &sz) == ERROR_SUCCESS) {
                char u8[256] = {};
                WideCharToMultiByte(CP_UTF8, 0, wname, -1, u8, sizeof u8, nullptr, nullptr);
                name = std::string(u8) + " (a custom plan)";
            }
            out = "power plan " + (name.empty() ? std::string("unknown") : name);
            LocalFree(scheme);
        }
        GUID ov{};
        if (get_overlay && get_overlay(&ov) == ERROR_SUCCESS) {
            static const GUID kEfficiency = {0x961cc777, 0x2547, 0x4f9d, {0x81, 0x74, 0x7d, 0x86, 0x18, 0x1b, 0x8a, 0x7a}};
            static const GUID kPerformance = {0xded574b5, 0x45a0, 0x4f42, {0x87, 0x37, 0x46, 0x34, 0x5c, 0x09, 0xc2, 0x38}};
            static const GUID kBetterBattery = {0x3af9b8d9, 0x7c97, 0x431d, {0xad, 0x78, 0x34, 0xa8, 0xbf, 0xea, 0x43, 0x9f}};
            const GUID kBalanced = {};
            const char* mode = IsEqualGUID(ov, kPerformance)      ? "best performance"
                               : IsEqualGUID(ov, kBalanced)       ? "balanced"
                               : IsEqualGUID(ov, kBetterBattery)  ? "better battery"
                               : IsEqualGUID(ov, kEfficiency)     ? "best power efficiency"
                                                                  : "other";
            out += std::string(out.empty() ? "" : ", ") + "power mode " + mode;
        }
        FreeLibrary(pp);
    }
    SYSTEM_POWER_STATUS ps{};
    if (GetSystemPowerStatus(&ps)) {
        const char* src = ps.ACLineStatus == 1 ? "on mains power" : ps.ACLineStatus == 0 ? "ON BATTERY" : "power source unknown";
        out += std::string(out.empty() ? "" : ", ") + src;
        if (ps.BatteryFlag != 128 && ps.BatteryFlag != 255 && ps.BatteryLifePercent <= 100) out += ", battery " + std::to_string(ps.BatteryLifePercent) + "%";
        if (ps.SystemStatusFlag & 1) out += ", BATTERY SAVER ON";
    }
    return out.empty() ? std::string("power plan unknown") : out;
}

Load sample() {
    Load l;
    const uint64_t wall = now_100ns();
    FILETIME c, e, k, u;
    uint64_t proc = 0;
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) proc = ft64(k) + ft64(u);
    FILETIME si, sk, su;
    uint64_t idle = 0, busy = 0;
    if (GetSystemTimes(&si, &sk, &su)) {
        idle = ft64(si);
        busy = ft64(sk) + ft64(su);   // kernel time includes the idle time
    }
    PROCESS_MEMORY_COUNTERS_EX pm{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pm), sizeof pm)) {
        l.working_set = pm.WorkingSetSize;
        l.committed = pm.PrivateUsage;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) l.ram_avail = ms.ullAvailPhys;
    DWORD handles = 0;
    if (GetProcessHandleCount(GetCurrentProcess(), &handles)) l.handles = handles;
    std::lock_guard<std::mutex> lk(g_mu);
    const int n = std::max(1, info().logical);
    if (g_wall && wall > g_wall) {
        const double dw = static_cast<double>(wall - g_wall);
        l.process_threads = static_cast<double>(proc - g_proc) / dw;
        l.process_pct = 100.0 * l.process_threads / n;
        const double db = static_cast<double>(busy - g_sys_busy);
        if (db > 0) l.system_pct = std::clamp(100.0 * (1.0 - static_cast<double>(idle - g_sys_idle) / db), 0.0, 100.0);
        l.valid = true;
    }
    g_wall = wall, g_proc = proc, g_sys_idle = idle, g_sys_busy = busy;
    return l;
}

std::vector<ThreadLoad> thread_loads(size_t max) {
    std::vector<ThreadLoad> out;
    std::vector<ThreadRecord> recs = diag_threads_dup();
    const uint64_t wall = now_100ns();
    std::lock_guard<std::mutex> lk(g_mu);
    const double dw = g_thread_wall && wall > g_thread_wall ? static_cast<double>(wall - g_thread_wall) : 0.0;
    std::unordered_map<uint32_t, uint64_t> next;
    for (const ThreadRecord& r : recs) {
        if (!r.host_handle) continue;
        FILETIME c, e, k, u;
        if (!GetThreadTimes(static_cast<HANDLE>(r.host_handle), &c, &e, &k, &u)) continue;
        const uint64_t t = ft64(k) + ft64(u);
        next[r.host_tid] = t;
        const auto it = g_thread_prev.find(r.host_tid);
        if (dw <= 0 || it == g_thread_prev.end() || t < it->second) continue;
        ThreadLoad tl;
        tl.name = r.name.empty() ? "(unnamed)" : r.name;
        tl.tid = r.host_tid;
        tl.pct = 100.0 * static_cast<double>(t - it->second) / dw;
        PROCESSOR_NUMBER pn{};
        if (GetThreadIdealProcessorEx(static_cast<HANDLE>(r.host_handle), &pn)) tl.ideal = pn.Group * 64 + pn.Number;
        out.push_back(std::move(tl));
    }
    g_thread_prev.swap(next);
    g_thread_wall = wall;
    diag_threads_release(recs);
    std::sort(out.begin(), out.end(), [](const ThreadLoad& a, const ThreadLoad& b) { return a.pct > b.pct; });
    if (out.size() > max) out.resize(max);
    return out;
}

}  // namespace nxrt::cpu
