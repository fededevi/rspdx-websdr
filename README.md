# RSPdx web receiver

A web receiver for an SDRplay RSPdx on Linux, for HF voice with live transcription. Radio amateurs and listeners can use it to follow many SSB channels at the same time.

## Contents

| Item | Description |
|---|---|
| `websdr/` | The web receiver: waterfall, detected channels, live audio, transcripts, conversation view for each frequency |
| `tools/stream.c` | Streams I/Q from the RSPdx as CS16, with a selectable IF filter. It changes its settings from stdin without a restart. |
| `tools/hamtext.py` | Whisper prompt for amateur radio, callsign parser and callsign validation |
| `tools/whisper_service.py` | Whisper service for a computer with an NVIDIA GPU |
| `tools/sstv/` | SSTV image detector and decoder for 8 kHz SSB audio, with its tests |
| `rspdx-websdr.sh`, `rspdx-websdr-stop.sh` | Start and stop the web receiver |

## Web receiver features

- One stream from the RSPdx gives all channels. An overlap-save FFT channelizer cuts each 8 kHz audio channel from the band.
- The span is selectable from 173 kHz to 6.8 MHz. Each span uses a matching IF filter.
- The receiver detects voice channels from the change of their level in time. It shows them on the spectrum and the waterfall.
- The carrier of each SSB channel goes to the 0.5 kHz grid, because most stations use a whole kHz or a 2.5 kHz grid. A tracker measures the edge of the voice band and corrects the carrier.
- Whisper transcribes the voice of all active channels. The number of channels changes with the load of the GPU.
- The parser finds callsigns in the spelled letters, for example "Lima Zulu 9 Whiskey" gives LZ9W. It validates them with the Super Check Partial list and the cty.dat prefix table.
- The conversation view shows a timeline for each frequency and the conversation in time order, with "Play all".
- The receiver decodes SSTV images on the SSTV frequencies in the view, for example 14.230 MHz. The page shows a live preview and a gallery. The decoder supports Martin M1/M2, Scottie S1/S2/DX, Robot 36/72 and PD modes. It measures the tuning error on the VIS header.

## Requirements

- Linux, an SDRplay RSPdx, and the SDRplay API 3.x with its service (`sdrplay`)
- SoapySDR and SoapySDRPlay3
- Python 3.10 or later, with the packages in `requirements.txt`
- Optional: a computer with an NVIDIA GPU (CUDA 12) for the transcription

The scripts look for the native programs and libraries in `local/bin` and `local/lib`. Install SoapySDR and SoapySDRPlay3 there, or change the paths.

## Build the native tools

1. Install SoapySDR and SoapySDRPlay3 into `local/`.
2. Build the streamer:

   ```
   gcc -O2 -o local/bin/rspdx-stream tools/stream.c -Ilocal/include -Llocal/lib -lSoapySDR
   ```

3. Install the Python packages:

   ```
   pip install -r requirements.txt
   ```

## Download the callsign data

The callsign validation uses two public files. Download them into `data/`:

```
mkdir -p data
curl -sSL -o data/MASTER.SCP https://www.supercheckpartial.com/MASTER.SCP
curl -sSL -o data/cty.dat https://www.country-files.com/cty/cty.dat
```

Download `MASTER.SCP` again before a contest, because the list changes.

## Start the web receiver

1. Stop the other programs that use the RSPdx.
2. Start the receiver, for example on the voice part of 20 m:

   ```
   ./rspdx-websdr.sh --center 14.225e6 --rate 384000 --antenna "Antenna C"
   ```

3. Open http://127.0.0.1:8490/ in a browser.
4. Click **Start audio**.
5. To stop the receiver, run `./rspdx-websdr-stop.sh`.

Run `python3 websdr/server.py --help` for all options. The page is bound to 127.0.0.1. If you bind it to another address, anyone on that network can open it, because it has no password.

## Use a GPU for the transcription

1. On the GPU computer, install Python, `faster-whisper`, `scipy`, and the CUDA 12 libraries `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`.
2. Start the service there:

   ```
   python whisper_service.py --model large-v3-turbo
   ```

   It listens on 127.0.0.1:8491 only. For a GPU of the Pascal generation, for example a GTX 1070, keep the default `--compute-type int8`.
3. On the receiver computer, open a shared SSH connection to the GPU computer:

   ```
   ssh -M -S ~/.ssh/cm-gpu -o ControlPersist=12h user@gpu-host
   ```

4. Write your settings in `local.env` next to the scripts. Git ignores this file.

   ```
   GPU_SSH=user@gpu-host
   ```

5. Start `./rspdx-websdr.sh`. It forwards the port through the SSH connection and uses the GPU. If the GPU service does not answer, the receiver uses its CPU.

## Data that stays local

The receiver writes these items into `logs/`:

- the WAV audio of each voice segment
- the transcripts
- the SSTV images

`.gitignore` permits only the named source files, so these items cannot go into the repository. The laws about the recording and the publication of radio communication are different in each country. Examine your local law before you record or publish received audio or text.

## License

MIT. See `LICENSE`.
