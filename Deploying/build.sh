#!/bin/bash
rm -rf *.so
echo "Programming language..."
command=$(ls|grep my_player)
py=$([[ $command =~ (^|[[:space:]])"my_player.py"($|[[:space:]]) ]] && echo 'yes' || echo 'no')
py3=$([[ $command =~ (^|[[:space:]])"my_player.py"($|[[:space:]]) ]] && echo 'yes' || echo 'no')
cpp=$([[ $command =~ (^|[[:space:]])"my_player.cpp"($|[[:space:]]) ]] && echo 'yes' || echo 'no')
c11=$([[ $command =~ (^|[[:space:]])"my_player11.cpp"($|[[:space:]]) ]] && echo 'yes' || echo 'no')
java=$([[ $command =~ (^|[[:space:]])"my_player.java"($|[[:space:]]) ]] && echo 'yes' || echo 'no')
if [ "$py" == "yes" ]; then
	cmd="python my_player.py"
	echo "PY"
elif [ "$py3" == "yes" ]; then
    cmd="python my_player.py"
	echo "PY3"
elif [ "$cpp" == "yes" ]; then
	g++ -O2 my_player.cpp -o exe
	cmd="./exe"
	echo "CPP"
elif [ "$java" == "yes" ]; then
	javac my_player.java
	cmd="java my_player"
	echo "JAVA"
elif [ "$c11" == "yes" ]; then
	g++ -std=c++0x -O2 my_player11.cpp -o exe
	cmd="./exe"
	echo "11"

else
    echo "ERROR: INVALID FILENAME"
	exit 1
fi

echo ""

prefix="./"
ta_agent=("random_player") # 1 TA players
surfix=".py"

# play funcion 
# only works for alias p
play()
{    
    echo Clean up... >&2
    echo Clean up... >&2 >> cresult.txt
    if [ -f "input.txt" ]; then
        rm input.txt
    fi
    if [ -f "output.txt" ]; then
        rm output.txt
    fi
    cp $prefix/init/input.txt ./input.txt

    echo Start Playing... >&2
    echo Start Playing... >&2 >>cresult.txt

	moves=0
	while true
	do
        if [ -f "output.txt" ]; then
	        rm output.txt
	    fi

        echo "Black makes move..." >&2
        echo "Black makes move..." >&2 >> cresult.txt
		eval "$1" >&2
		let moves+=1

		python $prefix/host.py -m $moves -v True >&2
		rst=$?

		if [[ "$rst" != "0" ]]; then
			break
		fi

        if [ -f "output.txt" ]; then
	        rm output.txt
	    fi

		echo "White makes move..." >&2
        echo "White makes move..." >&2 >> cresult.txt
		eval "$2" >&2
		let moves+=1

		python $prefix/host.py -m $moves -v True >&2
		rst=$?

		if [[ "$rst" != "0" ]]; then
			break
		fi
	done

	echo $rst
}

play_time=20

### start playing ###

echo ""
echo $(date)

for i in {0..0} # 1 TA players
do
    echo ""
    echo ==Playing with ${ta_agent[i]}==
    echo $(date)
    echo "" >>cresult.txt
    echo ==Playing with ${ta_agent[i]}== >>cresult.txt
    echo $(date) >>cresult.txt
    ta_cmd="python $prefix${ta_agent[i]}$surfix"
    black_win_time=0
    white_win_time=0
    black_tie=0
    white_tie=0
    for (( round=1; round<=$play_time; round+=2 )) 
    do
        # TA takes Black
        echo "=====Round $round====="
        echo "=====Round $round=====" >> cresult.txt
        echo Black:TA White:You 
        echo Black:TA White:You >> cresult.txt
        winner=$(play "$ta_cmd" "$cmd")
        if [[ "$winner" = "2" ]]; then
            echo 'White(You) win!'
            echo 'White(You) win!'>>cresult.txt
            let white_win_time+=1
        elif [[ "$winner" = "0" ]]; then
            echo Tie.
            let white_tie+=1
        else
            echo 'White(You) lose.'
        fi

        # Student takes Black
        echo "=====Round $((round+1))====="
        echo "=====Round $((round+1))=====" >> cresult.txt
        echo Black:You White:TA
        winner=$(play "$cmd" "$ta_cmd")
        if [[ "$winner" = "1" ]]; then
            echo 'Black(You) win!'
            echo 'Black(You) win!' >> cresult.txt
            let black_win_time+=1
        elif [[ "$winner" = "0" ]]; then
            echo Tie.
            let black_tie+=1
        else
            echo 'Black(You) lose.'
            echo 'Black(You) lose.'>> cresult.txt
        fi
    done


    echo =====Summary=====  
    echo "You play as Black Player | Win: $black_win_time | Lose: $((play_time/2-black_win_time-black_tie)) | Tie: $black_tie"
    echo "You play as White Player | Win: $white_win_time | Lose: $((play_time/2-white_win_time-black_tie)) | Tie: $white_tie"
    echo =====Summary=====  >> cresult.txt
    echo "You play as Black Player | Win: $black_win_time | Lose: $((play_time/2-black_win_time-black_tie)) | Tie: $black_tie" >>cresult.txt
    echo "You play as White Player | Win: $white_win_time | Lose: $((play_time/2-white_win_time-black_tie)) | Tie: $white_tie" >>cresult.txt
done

if [ -f "input.txt" ]; then
    rm input.txt
fi
if [ -f "output.txt" ]; then
    rm output.txt
fi
                                      
if [ -e "my_player.class" ]; then
    rm *.class
fi
if [ -e "exe" ]; then
    rm exe
fi
if [ -e "__pycache__" ]; then
    rm -rf __pycache__
fi
        
        
echo ""
echo Mission Completed.
echo $(date)