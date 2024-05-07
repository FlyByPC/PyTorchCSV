randomize timer
open "circle_1M.csv" for output as #1
dim as ulongint n
dim as double x,y,dx,dy,r
print #1, "X, Y, Q"
'print #1, "X, Y"
for n=1 to 1000000
   x=rnd*2
   y=rnd*2
   dx=x-1
   dy=y-1
   r=sqr(dx*dx+dy*dy)
   if r<0.8 then r=1 else r=0
   print #1,x;", ";y;", ";r
   'print #1,x;", ";y
   
   next n
close #1