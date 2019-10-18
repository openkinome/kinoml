set -euxo pipefail

# Temporarily change directory to $HOME to install software
pushd .
cd $HOME

# Install Miniconda
if [[ "$OPERATING_SYSTEM" == macOS* ]]; then
    # Make OSX md5 mimic md5sum from linux, alias does not work
    md5sum () {
        command md5 -r "$@"
    }
    MINICONDA=Miniconda3-latest-MacOSX-x86_64.sh
else
    MINICONDA=Miniconda3-latest-Linux-x86_64.sh
fi
MINICONDA_HOME=$HOME/miniconda
MINICONDA_MD5=$(curl -s https://repo.continuum.io/miniconda/ | grep -A3 $MINICONDA | sed -n '4p' | sed -n 's/ *<td>\(.*\)<\/td> */\1/p')
wget -q https://repo.continuum.io/miniconda/$MINICONDA
if [[ $MINICONDA_MD5 != $(md5sum $MINICONDA | cut -d ' ' -f 1) ]]; then
    echo "Miniconda MD5 mismatch"
    exit 1
fi
bash $MINICONDA -b -p $MINICONDA_HOME

# Configure miniconda
export PIP_ARGS="-U"
# Enable Miniconda
eval "$($MINICONDA_HOME/bin/conda shell.bash hook)"
conda activate

conda config --add channels conda-forge

conda config --set always_yes yes
conda install conda conda-build jinja2 anaconda-client
conda update --quiet --all

cat << "EOF" > ~/.minicondarc
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
if [ -d $HOME/miniconda/envs/test ]; then
    conda activate test
else
    conda activate
fi
EOF

# Restore original directory
popd
