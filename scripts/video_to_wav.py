import subprocess
import os

def main():
    # listing the files in the current directory
    files = os.listdir("./")
    for f in files:
        # filtering the video files
        if f.lower()[-3:] == "mp4":
            infile = f
            # setting output file format
            outfile = f[:-3] + "wav"
            command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn  {}".format(infile, outfile)
            # calling a sub-rotuine to execute the command
            subprocess.call(command, shell=True)

main()

