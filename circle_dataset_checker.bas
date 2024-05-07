dim as double x,y,dx,dy,r

open "circle_1M_labeled.csv" for INPUT as #1

dim as ulongint passes,fails,total
passes=0
fails=0
total=0


while not eof(1)
   input #1,x,y,r
   total=total+1
   dx=x-1
   dy=y-1
   if sqr(dx*dx+dy*dy)<0.8 then
      if r=1 then passes=passes+1
      if r=0 then fails=fails+1
   else
      if r=0 then passes=passes+1
      if r=1 then fails=fails+1
      end if
   wend

print passes, fails, total, 100*(passes/total);"%"

close #1

sleep
