#!/usr/bin/env/python
##============================================================



##============================================================
## Returns the f%-quantile of a list.

def quantile(lst, f=0.5):
    l,r = 0,256
    targetCount = int(len(lst)*f)
    while l+1 < r:
        m = (l+r)/2
        count = 0
        for x in lst:
            if x < m: count += 1
        if count <= targetCount:
            l = m
        else:
            r = m
    return l
    
##============================================================
## Returns the f%-quantile of a list. Uses histogram method.

def quantile2(lst, f=0.5):
    ## build histogram    
    h = [0]*256
    for x in lst:
        h[x] = h[x]+1

    ## find median value in histogram
    sum = 0
    targetSum = int(len(lst)*f)
    for i in range(0, 255):
        sum += h[i]
        if (sum > targetSum):
            return i

##============================================================

def histquantile_discrete(hist,f=0.5):
  count = 0
  targetSum = int(sum(hist)*f)
  for i in range(0,256):
      count += hist[i]
      if (count > targetSum):
          return i

##============================================================

def histquantile(hist,bins,f=0.5):
  count = 0
  targetSum = int(sum(hist)*f)
  for i in range(len(bins)):
      count += hist[i]
      if (count > targetSum):
          return bins[i]
          
##============================================================
if __name__=="__main__":
	main()
