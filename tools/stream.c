/* Stream I/Q from the RSPdx to stdout as CS16, with a chosen IF filter.
 * Usage: stream freq_hz rate_hz bw_hz [antenna [settings]]   (bw 0 = driver default)
 *        settings: "key=value,..." with the driver keys (rfgain_sel, rfnotch_ctrl, ...)
 *        and agc=0|1 and ifgr=dB
 *        stream --list                              (sample rates and filters)
 * While it runs, a line "freq_hz rate_hz bw_hz" on stdin changes the settings
 * without closing the device, and a line "set key=value,..." changes the settings above. */
#include <SoapySDR/Device.h>
#include <SoapySDR/Formats.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile int run = 1;

static void apply(SoapySDRDevice *dev, const char *list)
{
    char buf[400];
    snprintf(buf, sizeof buf, "%s", list);
    for (char *kv = strtok(buf, ", \n"); kv; kv = strtok(NULL, ", \n")) {
        char *v = strchr(kv, '=');
        if (!v) continue;
        *v++ = 0;
        if (!strcmp(kv, "agc")) SoapySDRDevice_setGainMode(dev, SOAPY_SDR_RX, 0, atoi(v) != 0);
        else if (!strcmp(kv, "ifgr")) SoapySDRDevice_setGainElement(dev, SOAPY_SDR_RX, 0, "IFGR", atof(v));
        else SoapySDRDevice_writeSetting(dev, kv, v);
    }
}
static void stop(int s) { (void)s; run = 0; }

int main(int argc, char **argv)
{
    SoapySDRKwargs args = {0};
    SoapySDRKwargs_set(&args, "driver", "sdrplay");
    SoapySDRDevice *dev = SoapySDRDevice_make(&args);
    if (!dev) { fprintf(stderr, "no device: %s\n", SoapySDRDevice_lastError()); return 1; }
    if (argc > 1 && !strcmp(argv[1], "--list")) {
        size_t n;
        double *r = SoapySDRDevice_listSampleRates(dev, SOAPY_SDR_RX, 0, &n);
        printf("rates:");
        for (size_t i = 0; i < n; i++) printf(" %.0f", r[i]);
        double *b = SoapySDRDevice_listBandwidths(dev, SOAPY_SDR_RX, 0, &n);
        printf("\nbandwidths:");
        for (size_t i = 0; i < n; i++) printf(" %.0f", b[i]);
        printf("\n");
        SoapySDRDevice_unmake(dev);
        return 0;
    }
    if (argc < 4) { fprintf(stderr, "usage: %s freq_hz rate_hz bw_hz [antenna] | --list\n", argv[0]); return 1; }
    signal(SIGINT, stop);
    signal(SIGTERM, stop);
    signal(SIGPIPE, stop);
    SoapySDRDevice_setAntenna(dev, SOAPY_SDR_RX, 0, argc > 4 ? argv[4] : "Antenna A");
    SoapySDRDevice_setSampleRate(dev, SOAPY_SDR_RX, 0, atof(argv[2]));
    if (atof(argv[3]) > 0)
        SoapySDRDevice_setBandwidth(dev, SOAPY_SDR_RX, 0, atof(argv[3]));
    SoapySDRDevice_setFrequency(dev, SOAPY_SDR_RX, 0, atof(argv[1]), NULL);
    SoapySDRDevice_writeSetting(dev, "biasT_ctrl", "false");
    if (argc > 5) apply(dev, argv[5]);
    fprintf(stderr, "rate %.0f, bandwidth %.0f, frequency %.0f\n",
            SoapySDRDevice_getSampleRate(dev, SOAPY_SDR_RX, 0),
            SoapySDRDevice_getBandwidth(dev, SOAPY_SDR_RX, 0),
            SoapySDRDevice_getFrequency(dev, SOAPY_SDR_RX, 0));
    /* the settings of the receiver, for the log */
    const char *keys[] = {"rfgain_sel", "agc_setpoint", "hdr_ctrl", "rfnotch_ctrl", "dabnotch_ctrl", "biasT_ctrl", "iqcorr_ctrl"};
    fprintf(stderr, "antenna %s, AGC %s, gain %.0f dB (IFGR %.0f, RFGR %.0f)",
            SoapySDRDevice_getAntenna(dev, SOAPY_SDR_RX, 0),
            SoapySDRDevice_getGainMode(dev, SOAPY_SDR_RX, 0) ? "on" : "off",
            SoapySDRDevice_getGain(dev, SOAPY_SDR_RX, 0),
            SoapySDRDevice_getGainElement(dev, SOAPY_SDR_RX, 0, "IFGR"),
            SoapySDRDevice_getGainElement(dev, SOAPY_SDR_RX, 0, "RFGR"));
    for (size_t i = 0; i < sizeof keys / sizeof keys[0]; i++)
        fprintf(stderr, ", %s=%s", keys[i], SoapySDRDevice_readSetting(dev, keys[i]));
    fprintf(stderr, "\n");
    SoapySDRStream *rx = SoapySDRDevice_setupStream(dev, SOAPY_SDR_RX, SOAPY_SDR_CS16, NULL, 0, NULL);
    if (!rx) { fprintf(stderr, "no stream: %s\n", SoapySDRDevice_lastError()); return 1; }
    SoapySDRDevice_activateStream(dev, rx, 0, 0, 0);
    size_t mtu = SoapySDRDevice_getStreamMTU(dev, rx);
    short *buf = malloc(4 * mtu);
    struct pollfd in = {0, POLLIN, 0};
    char line[200];
    long overflows = 0;
    while (run) {
        if (poll(&in, 1, 0) > 0) {
            double f, r, bw;
            if (!fgets(line, sizeof line, stdin)) { in.fd = -1; continue; }
            if (!strncmp(line, "set ", 4)) {
                apply(dev, line + 4);
                fprintf(stderr, "set %s", line + 4);
            } else if (sscanf(line, "%lf %lf %lf", &f, &r, &bw) == 3) {
                if (r != SoapySDRDevice_getSampleRate(dev, SOAPY_SDR_RX, 0) || bw != SoapySDRDevice_getBandwidth(dev, SOAPY_SDR_RX, 0)) {
                    SoapySDRDevice_deactivateStream(dev, rx, 0, 0);
                    SoapySDRDevice_setSampleRate(dev, SOAPY_SDR_RX, 0, r);
                    if (bw > 0)
                        SoapySDRDevice_setBandwidth(dev, SOAPY_SDR_RX, 0, bw);
                    SoapySDRDevice_setFrequency(dev, SOAPY_SDR_RX, 0, f, NULL);
                    SoapySDRDevice_activateStream(dev, rx, 0, 0, 0);
                } else {
                    SoapySDRDevice_setFrequency(dev, SOAPY_SDR_RX, 0, f, NULL);
                }
                fprintf(stderr, "rate %.0f, bandwidth %.0f, frequency %.0f\n",
                        SoapySDRDevice_getSampleRate(dev, SOAPY_SDR_RX, 0),
                        SoapySDRDevice_getBandwidth(dev, SOAPY_SDR_RX, 0),
                        SoapySDRDevice_getFrequency(dev, SOAPY_SDR_RX, 0));
            }
        }
        void *b[] = {buf};
        int flags; long long t;
        int n = SoapySDRDevice_readStream(dev, rx, b, mtu, &flags, &t, 200000);
        if (n == SOAPY_SDR_OVERFLOW) { if (++overflows % 10 == 1) fprintf(stderr, "overflow %ld\n", overflows); continue; }
        if (n <= 0) continue;
        if (fwrite(buf, 4, n, stdout) != (size_t)n) break;
    }
    SoapySDRDevice_deactivateStream(dev, rx, 0, 0);
    SoapySDRDevice_closeStream(dev, rx);
    SoapySDRDevice_unmake(dev);
    return 0;
}
