import os
import subprocess
import speech_recognition as sr
import wave
import contextlib

def main():
    c =0
    # listing the files in the current director
    files = os.listdir("./")
    r = sr.Recognizer()
    rec_f = 0
    for f in files:
        rec_f +=1
        if f.lower()[-3:] == "wav":
            j = 0
            k = 0
            fname = f
            # finding the duration of the audio file
            with contextlib.closing(wave.open(fname,'r')) as fn :
                frames = fn.getnframes()
                rate = fn.getframerate()
                duration = frames / float(rate)
                tr =0
            for i in range(int(duration)):
                if j < int(duration) - 6:
                    j = j+k
                    k = 15
                    infile = f
                    # setting the output file name and format
                    outfile = str(c) + ".wav"
                    # command to trim the audio file to 15sec
                    command = "ffmpeg -i {} -ss {} -t {} -async 1 {}".format(infile, j, k, outfile)
                    # calling a sub-routine to execute the command
                    subprocess.call(command, shell=True)
                    tr +=1
                    print("\ntrimmed  ", tr, " part\n")
                    c +=1
                    # recording the audio in the trimmed file
                    with sr.AudioFile(outfile) as source:
                        audio = r.record(source)
                    # using try except block to handle other exceptions
                    try:
                        print("\nstarted recognising  :", rec_f, "file \n")
                        fname = f.lower()[:-3] + 'txt'
                        with open(fname, 'a+') as tp:
                            # sending the recording to Google and appending the responce to a text file
                            written_len = tp.write(" " + r.recognize_google(audio))

                    except sr.UnknownValueError:            
                        print("Google Speech Recognition could not understand audio") 
    
                    except sr.RequestError as e: 
                        print("Could not request results from Google Speech Recognition service; {0}".format(e)) 


main()
