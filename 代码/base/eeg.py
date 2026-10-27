from scipy.signal import welch
from scipy.integrate import simpson
import numpy as np
import mne
from scipy import signal
import math


class GenericEegController(object):

    def __init__(self, filename, config):
        self.filename = filename
        self.buffer_sec = config['buffer_sec']
        self.frequency = config['sampling_frequency']
        self.window_sec = config['window_sec']
        self.step = int(config['hop_sec'] * self.frequency)
        self.interest_bands = config['interest_bands']
        self.f_trans_interest_bands = config['f_trans_interest_bands']
        self.channel_slice = config['channel_slice']
        self.eeg_feature_list = config['features']
        self.filter_type = config['filter_type']
        self.filter_order = config['filter_order']
        self.extracted_data = self.preprocessing()

    def calculate_bandpower(self, data):
        data_np = data[:].T
        power_spectram_densities = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            seg_data = data_np[:, start:end]
            bandpower = bandpower_multiple(seg_data, sampling_frequence=self.frequency,
                                           band_sequence=self.interest_bands, relative=True)
            power_spectram_densities.append(bandpower)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        power_spectram_densities = np.stack(power_spectram_densities)
        return power_spectram_densities

    def calculate_eeg(self, data):
        data_np = data[:].T
        power_spectram_densities = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            eeg_seg = data_np[:, start:end]
            eeg_seg = np.concatenate(eeg_seg)
            power_spectram_densities.append(eeg_seg)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        power_spectram_densities = np.stack(power_spectram_densities)
        return power_spectram_densities

    def calculate_DE(self, data):
        data_np = data[:].T
        data_filtered = filter_band(
            data=data_np, bands=self.interest_bands, stop_fre=self.f_trans_interest_bands,
            filter_type=self.filter_type, fs=self.frequency, order=self.filter_order
        )
        DEs = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            des = []
            for f in data_filtered:
                de = DE(f[:, start:end])
                des.append(de)
            des = np.stack(des).T
            des_flat = np.reshape(des, (-1,))
            DEs.append(des_flat)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        DEs = np.stack(DEs)
        return DEs

    def calculate_fHjorth_per_fre(self, data):
        data_np = data[:].T
        data_filtered = filter_band(
            data=data_np, bands=self.interest_bands, stop_fre=self.f_trans_interest_bands,
            filter_type=self.filter_type, fs=self.frequency, order=self.filter_order
        )
        Hjorths = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            hjorth = []
            for f in data_filtered:
                de = TemporalFeature(f[:, start:end])
                hjorth.append(de)
            hjorth = np.transpose(np.stack(hjorth), (1, 0, 2))
            hj_flat = np.reshape(hjorth, (-1,))
            Hjorths.append(hj_flat)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        Hjorths = np.stack(Hjorths)
        Hjorths = (Hjorths - np.mean(Hjorths, axis=-1, keepdims=True)) / np.std(Hjorths, axis=-1, keepdims=True)
        return Hjorths

    def calculate_fHjorth(self, data):
        data_np = data[:].T
        Hjorths = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            tmf = TemporalFeature(data_np[:, start:end])
            tmf_flat = np.reshape(tmf, (-1,))
            Hjorths.append(tmf_flat)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        Hjorths = np.stack(Hjorths)
        return Hjorths

    def calculate_RP(self, data):
        data_np = data[:].T
        data_filtered = filter_band(
            data=data_np, bands=self.interest_bands, stop_fre=self.f_trans_interest_bands,
            filter_type=self.filter_type, fs=self.frequency, order=self.filter_order
        )
        RPs = []
        start = 0
        end = int(start + self.frequency * self.window_sec)
        while end < data_np.shape[1]:
            rp_t = RP(data_filtered[:, :, start:end])
            rp_flat = np.reshape(rp_t, (-1,))
            RPs.append(rp_flat)
            start = start + self.step
            end = int(start + self.frequency * self.window_sec)
        RPs = np.stack(RPs)
        RPs = RPs / np.sum(RPs, axis=-1, keepdims=True)
        return RPs

    def preprocessing(self):
        raw_data = self.read_data()
        channel_type_dictionary = self.get_channel_type_dictionary(raw_data)
        raw_data = self.set_channel_types_from_dictionary(raw_data, channel_type_dictionary)
        crop_range = self.get_crop_range_in_second(raw_data)
        cropped_raw_list = self.crop_data(raw_data, crop_range)

        all_extracted = {}

        for cropped_raw_data in cropped_raw_list:
            cropped_eeg_raw_data = self.get_eeg_data(cropped_raw_data)
            filtered_data_np = self.filter_eeg_data(cropped_eeg_raw_data)
            average_referenced_data_np = self.average_reference(filtered_data_np)

            extracted_data = {}
            if "eeg_raw" in self.eeg_feature_list:
                eeg_segments = self.calculate_eeg(average_referenced_data_np)
                extracted_data.update({'eeg_raw': eeg_segments})

            if "eeg_bandpower" in self.eeg_feature_list:
                bandpower = self.calculate_bandpower(average_referenced_data_np)
                extracted_data.update({'eeg_bandpower': bandpower})

            if "eeg_DE" in self.eeg_feature_list:
                bandpower = self.calculate_DE(average_referenced_data_np)
                extracted_data.update({'eeg_DE': bandpower})

            if "eeg_RP" in self.eeg_feature_list:
                bandpower = self.calculate_RP(average_referenced_data_np)
                extracted_data.update({'eeg_RP': bandpower})

            if "eeg_Hjorth" in self.eeg_feature_list:
                bandpower = self.calculate_fHjorth(average_referenced_data_np)
                extracted_data.update({'eeg_Hjorth': bandpower})

            for k, v in extracted_data.items():
                if k not in all_extracted:
                    all_extracted[k] = v
                else:
                    all_extracted[k] = np.concatenate([all_extracted[k], v], axis=0)

        for key in all_extracted:
            all_extracted[key] = all_extracted[key].astype(np.float32)
            print(f"[DEBUG] {key} shape: {all_extracted[key].shape}")

        return all_extracted

    def produce_bandpower_array(self, band_power):
        num_bp_per_sample = len(self.interest_bands) * 32
        bp_array = np.reshape(band_power, (-1, num_bp_per_sample))
        return bp_array

    def set_interest_bands(self):
        interest_bands = [(0.3, 4), (4, 8), (8, 12), (12, 18), (18, 30), (30, 45)]
        return interest_bands

    def filter_eeg_data(self, data):
        filtered_eeg_data = data.copy().load_data().filter(l_freq=0.3, h_freq=45, method='iir')
        return filtered_eeg_data

    def average_reference(self, data):
        average_referenced_data = data.copy().load_data().set_eeg_reference()
        return average_referenced_data[:][0].T

    def read_data(self):
        raw_data = mne.io.read_raw_bdf(self.filename)
        return raw_data

    def get_channel_slice(self):
        channel_slice = {'eeg': slice(0, 32), 'ecg': slice(32, 35), 'misc': slice(35, -1)}
        return channel_slice

    def get_channel_type_dictionary(self, data):
        channel_type_dictionary = {}
        for modal, slicing in self.channel_slice.items():
            channel_type_dictionary.update({channel: modal
                                            for channel in data.ch_names[slicing]})
        return channel_type_dictionary

    def set_channel_types_from_dictionary(self, data, channel_type_dictionary):
        data = data.set_channel_types(channel_type_dictionary)
        return data

    def get_crop_range_in_second(self, data):
        crop_range = [[30. - self.window_sec / 2, data.times.max() - 30 + self.buffer_sec]]
        return crop_range

    def crop_data(self, data, crop_range):
        cropped_data = []
        for start, end in crop_range:
            cropped_data.append(data.copy().crop(tmin=start, tmax=end))
        return cropped_data

    def get_eeg_data(self, data):
        eeg_data = data.copy().pick_types(eeg=True)
        return eeg_data


def bandpower_multiple(data, sampling_frequence, band_sequence, window_sec=1, relative=False):
    n_chan, n_time = data.shape
    nperseg = int(window_sec * sampling_frequence)
    band_powers_all = []
    for ch in range(n_chan):
        sig = data[ch, :]
        freqs, welch_psd = welch(sig, fs=sampling_frequence, nperseg=nperseg)
        freq_res = freqs[1] - freqs[0]
        band_p = []
        for (low, high) in band_sequence:
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = simpson(welch_psd[idx], dx=freq_res)
            if relative:
                band_power /= simpson(welch_psd, dx=freq_res)
            band_p.append(band_power)
        band_powers_all.append(band_p)
    return np.array(band_powers_all)


def filter_band(data, bands, stop_fre, filter_type, fs, order):
    filtered_data = []
    for idx, f in enumerate(bands):
        stop_f = stop_fre[idx]
        if filter_type == 'cheby2':
            data_ = bandpassfilter_cheby2_sos(
                data=data, bandFiltCutF=[f[0], f[1]], fs=fs, filtAllowance=[stop_f[0], stop_f[1]], axis=1,
            )
        else:
            data_ = band_pass_butter(
                data=data, f_low=f[0], f_high=f[1], fs=fs, filter_order=order
            )
        filtered_data.append(data_)
    return np.stack(filtered_data)


def bandpassfilter_cheby2_sos(data, bandFiltCutF=[0.3, 40], fs=256, filtAllowance=[0.2, 5], axis=2):
    aStop = 40
    aPass = 1
    nFreq = fs / 2
    if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] is None or bandFiltCutF[1] >= fs / 2.0):
        print("Not doing any filtering. Invalid cut‑off specifications")
        return data
    elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
        fPass = bandFiltCutF[1] / nFreq
        fStop = (bandFiltCutF[1] + filtAllowance[1]) / nFreq
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, fStop, 'lowpass', output='sos')
    elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
        fPass = bandFiltCutF[0] / nFreq
        fStop = (bandFiltCutF[0] - filtAllowance[0]) / nFreq
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, fStop, 'highpass', output='sos')
    else:
        fPass = (np.array(bandFiltCutF) / nFreq).tolist()
        fStop = [(bandFiltCutF[0] - filtAllowance[0]) / nFreq, (bandFiltCutF[1] + filtAllowance[1]) / nFreq]
        [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, fStop, 'bandpass', output='sos')
    return signal.sosfilt(sos, data, axis=axis)


def band_pass_cheby2(data, f_low, f_high, fs, filter_order):
    f_trans = 2
    f_pass = np.asarray([f_low, f_high])
    f_stop = np.asarray([f_low - f_trans, f_high + f_trans])
    gpass = 3
    gstop = 30
    f_nqs = fs / 2
    wp = f_pass / f_nqs
    ws = f_stop / fs
    order = filter_order
    b, a = signal.cheby2(order, gstop, ws, btype='bandpass')
    data_filtered = signal.lfilter(b, data)
    return data_filtered


def band_pass_butter(data, f_low, f_high, fs, filter_order=8):
    wn1 = 2 * f_low / fs
    wn2 = 2 * f_high / fs
    b, a = signal.butter(filter_order, [wn1, wn2], 'bandpass')
    filtered = signal.filtfilt(b, data)
    return filtered


def TemporalFeature(data):
    feature = []
    for chan in data:
        diff = np.diff(chan)
        ddiff = np.diff(diff)
        var = np.var(chan)
        dvar = np.var(ddiff)
        activity = var
        mobility = np.sqrt(dvar / var)
        complexity = np.sqrt(np.var(ddiff) / mobility)
        feature.append([activity, mobility, complexity])
    feature = np.stack(feature, axis=0)
    return feature


def DE(data):
    de_val = 0.5 * np.log(2 * np.pi * np.sqrt(np.std(data, axis=-1)))
    return de_val


def RP(data):
    data_temp = np.power(data, 2)
    data_temp = np.sum(data_temp, axis=-1)
    power_sum = np.sum(data_temp, axis=0, keepdims=True)
    data_temp = data_temp / power_sum
    return data_temp