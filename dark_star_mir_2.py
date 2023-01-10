#!/usr/bin/env python
# coding: utf-8

# In[1]:


#adapted from https://musicinformationretrieval.com/
# see also http://www.notesartstudio.com/about.html
import numpy
import scipy
import pandas
import sklearn
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd


# In[3]:


x, sr = librosa.load("dark_star.wav")
# this version of dark star from https://www.dead.net/30daysofdead


# In[4]:


print(x.shape)


# In[5]:


print(sr)


# In[11]:


plt.figure(figsize=(14,5))
librosa.display.waveplot(x[:5000000],sr=sr) # overflows with full length


# In[12]:


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14,5))
librosa.display.specshow(Xdb,sr=sr,x_axis='time',y_axis='hz')


# In[20]:


k = 1
ipd.Audio(x[::k],rate=sr/k)


# In[21]:


X = scipy.fft(x[:20*sr])
X_mag= numpy.absolute(X)
f = numpy.linspace(0,sr,4096)
plt.figure(figsize=(14,5))
plt.plot(f[:2000],X_mag[:2000])
plt.xlabel("Frequency(Hz)")


# In[27]:


hop_length = 512
chromagram = librosa.feature.chroma_cqt(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')


# In[28]:


chromagram = librosa.feature.chroma_cens(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')


# In[34]:


spectral_centroids = librosa.feature.spectral_centroid(x+.001, sr=sr)[0]
spectral_centroids.shape


# In[35]:


frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)


# In[36]:


def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# In[37]:


librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r') # normalize for visualization purposes


# In[38]:


spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))


# In[39]:


spectral_contrast = librosa.feature.spectral_contrast(x, sr=sr)
spectral_contrast.shape


# In[40]:


plt.imshow(normalize(spectral_contrast, axis=1), aspect='auto', origin='lower', cmap='coolwarm')


# In[41]:


spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')


# In[ ]:


# Because the autocorrelation produces a symmetric signal, we only care about the "right half".
r = numpy.correlate(x, x, mode='full')[len(x)-1:]
print(x.shape, r.shape)

plt.figure(figsize=(14, 5))
plt.plot(r[:10000])
plt.xlabel('Lag (samples)')
plt.xlim(0, 10000)

r = librosa.autocorrelate(x, max_size=10000)
print(r.shape)
plt.figure(figsize=(14, 5))
plt.plot(r)
plt.xlabel('Lag (samples)')
plt.xlim(0, 10000)

#more to do here

