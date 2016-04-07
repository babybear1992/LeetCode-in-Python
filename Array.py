#283 # Move Zeroes
class Solution:
    def moveZeroes(self,nums):
        L = len(nums)
        i = 0
        while i < L:
            if nums[i] == 0:
                nums.append(nums.pop(i))
                L -= 1
            else:
                i += 1
# solution 2
    def moveZeroes(self,nums):
        total = nums.count(0)
        for i in range(total):
            nums.remove(0)
            nums.append(0)
# solution 3
    def moveZeroes(self,nums):
        y = 0
        for x in range(len(nums)):
            if nums[x]:
                nums[x], nums[y] = nums[y], nums[x]
                y += 1

73 # Set Matrix Zeroes
class Solution:
    def setZeroes(self,matrix):
        row = len(matrix)
        col = len(matrix[0])
        rowToZeroes = set()
        colToZeroes = set()
        for i in xrange(row):
            for j in xrange(col):
                if matrix[i][j] == 0:
                    rowToZeroes.add(i)
                    colToZeroes.add(j)
        for i in rowToZeroes:
            for j in xrange(col):
                matrix[i][j] = 0

        for j in colToZeroes:
            for i in xrange(row):
                matrix[i][j] = 0
        

74 # Search 2D Matrix
class Solution:
    def searchMatrix(self,matrix,target):
        i = 0 # search from first row
        j = len(matrix[0] - 1) # search from last column
        while i <= len(matrix) - 1 and j >= 0:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            else: # matrix[i][j] < target
                i += 1
        return False

240 # Search 2D Matrix II
class Solution:
    def searchMatrix(self,matrix,target):
        i = 0 # seach from first row
        j = len(matrix[0]) - 1 # sarch from last column
        for i in range(len(matrix) - 1):
            while j and matrix[i][j] > target:
                j -= 1
                if matrix[i][j] == target:
                    return True
        return False
# or use the follwoing while loop
        while i <= len(matrix) - 1 and j >= 0:
            if matrix[i][j] > target:
                j -= 1
            elif matrix[i][j] < target:
                i += 1
            else:
                return Truie
        return False

54 # Spiral Matrix
class Solution:
    def sparialOrder(self,matrix):
         if matrix == []:
            return []
        # 矩阵四个角的index
        left = 0
        right = len(matrix[0]) - 1
        up = 0
        down = len(matrix) - 1

        res = []
        direct = 0
        while True:
            if direct == 0:
                for i in range(left,right+1):
                    res.append(matrix[up][i])
                up += 1
            if direct == 1:
                for i in range(up,down+1):
                    res.append(matrix[i][right])
                right -= 1
            if direct == 2:
                for i in range(right,left-1,-1):
                    res.append(matrix[down][i])
                down -= 1
            if direct == 3:
                for i in range(down,up-1,-1):
                    res.append(matrix[i][left])
                left += 1
            if up > down or left > right:
                return res
        direct = (direct+1) % 4

59 # Spiral Matrix II
class Solution:
    def generateMatrix(self,n):
        if n < 0:
            return False
        if n == 0:
            return []
        # build the empty matrix
        matrix = [[0 for i in xrange(n)] for j in xrange(n)]
        
        direction = 0
        # 矩阵4个角的坐标
        count = 1
        up = 0 
        down = n - 1
        right = n-1
        left = 0
        
        while True:
            if direction == 0:
                for i in range(left,right+1):
                    matrix[up][i] = count
                    count += 1
                up += 1
            if direction == 1:
                for i in range(up,down+1):
                    matrix[i][right] = count
                    count += 1
                right -= 1
            if direction == 2:
                for i in range(right,left-1,-1):
                    matrix[down][right] = count
                    count += 1
                down -= 1

            if direction == 3:
                for i in range(down,up-1,-1):
                    matrix[i][left] = count
                    count += 1
                left += 1

            if up < down and left > right:
                return matrix
            direction = (direction + 1) % 4

48 # Rotate Image
class Solution:
    def rotate(self,matrix):
        size = len(matrix)
        # 顺时针45度交换元素
        for i in xrange(size):
            for j in xrange(size-1-i):
                matrix[i][j],matrix[size-1-j][size-1-i] = matrix[size-1-j][size-1-i],matrix[i][j]
        # 以row = 2/size 为轴，上下翻转
        for i in xrange(size/2):
            for j in xrange(size):
                matrix[i][j], matrix[size-1-i][size-1-j] = matrix[size-1-j][size-1-i],matrix[i][j]

# solution 2
    def rotate(self,matrix):
        matrix[::] = zip(*matrix[::-1])


35 # Search Insert Position
class Solution:
    def searcInsert(self,nums,target):
        left = 0
        right = len(nums) - 1
        while left <= right:
            center = (left+right)/2
            if nums[center] == target:
                return center
            elif nums[center] > target:
                right = center - 1
            else: # nums[center] < target
                left = center + 1
        return left

27 # Remove Element
class Solution:
    def removeElement(self,nums,val):
        while val in nums:
            nums.remove(val)
        return len(nums)

26 # Remove Duplicates from Sorted Array
class Solution:
    def removeDuplicates(self,nums):
        j = 0
        if len(nums) == 0:
            return None
        for i in range(len(nums)):
            if nums[i] != nums[j]:
                nums[i], nums[j+1] = nums[j+1], nums[i]
                j =  j+ 1
        return j + 1
# solution 2 
    def removeDuplicates(self,nums):
        if len(nums) == 0:
            return None
        i = 0
        for x in nums:
            if nums[i] != x:
                i += 1
                nums[i] = x
        return i + 1
189 # Rotate Array
class Solution:
    def rotate(self,nums,k):
        n = len(nums)
        if k > 0 and n > 1:
            nums[:] = nums[n-k:] + nums[:,n-k]
# solution 2
    def rotate(self,nums,k):
        n = len(nums)
        index = 0
        distance = 0
        cur = nums[0]
        for i in range(n):
            index = (index + k ) % n
            nums[index], cur = cur, nums[index]

            distance = (distance+k) % n
            if distance == 0:
                index = (index + 1( % n
                cur = nums[index]

# solution 3
    def rotate(self,nums,k):
        n = len(nums)
        k = k % n
        self.reverse(nums,0,n-k)
        self.reverse(nums,n-k,n)
        self.reverse(nums,0,n)
    def reverse(self,nums,start,end):
        for x in range(start,(start+end)/2):
            nums[x] ^= nums[start+end-x-1]
            nums[start+end-x-1]^= nums[x]
            nums[x]^= nums[start+end-x-1]

88 # Merge Sorted Array
class Solution:
    def merge(self, nums1,m,nums2,n):
        i = m - 1
        j = n - 1
        k = m - n + 1
        while i >=0 and j >= 0:
            if nums2[j] > nums1[i]:
                nums1[k] = nums2[2]
                j -= 1
            else:
                nums1[k] = nums1[i]
            k -= 1
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1
75 # Sort Colors
class Solution:
    def sortColors(self,nums):
        if nums = []:
            return None
        p0 = 0
        p2 = len(nums) - 1
        while i <= p2:
            if nums[i] == 0:
                nums[i],nums[p0] = nums[p0], nums[i]
                p0 += 1
                i += 1
            elif nums[i] == 2:
                nums[i],nums[p2] = nums[p2],nums[i]
                p2 -= 1
            else: # nums[i] == 1
                i += 1
169 # Majority Element
class Solution:
    def majorityElement(self,nums):
        candidate = None
        count = 0
        for i in nums:
            if count == 0:
                candidate = i
                count += 1
            elif i == candidate:
                count += 1
            else:
                count -= 1
        return candidate

118 # Pascal's Traginle
class Solution:
    def generate(self,numRows):
        if numRows == 0:
            return []
        if numRows == 1:
            return [[1]]
        if numRows == 2:
            return [[1],[1,1]]
        elif numRows > 2:
            list = [[] for i in range(numRows)]
            for i in range(0,numRows):
                list[i] = [1 for j in range(i+1)]
            for i in range(2,numRows):
                for j in range(1,i):
                    list[i][j] = list[i-1][j-1] + list[i-1][j]
        return list

119 # Pascal's Triangle II
class Solution:
    def generate(self,rowIndex):
        if rowIndex == 0:
            return []
        if rowIndex == 1:
            return [1,1]
        list = [[] for i in range(rowIndex+1)]
        list[0] = []
        list[1] = [1,1]
        for i in range(2,rowIndex+1):
            list[i] = [1 for j in range(i+1)]
            for j in xrange(1,i):
                list[i][j] = list[i-1][j-1] + list[i-1][j]
        return list[rowIndex]

153 # Find Minimum in Rotated Sorted Array
class Solution:
    def findMin(self,nums):
        ans = nums[0]
        low = 0
        high = len(nums) - 1
        while low <= high:
            center = (low + high) / 2
            if nums[mid] <= nums[high]:
                high  = mid - 1
            else:
                low = mid + 1
        ans = min(nums[mid],ans)

33 # Search in Rotated Sorted Array
class Solution:
    def search(self,nums,target):
        left = 0
        right = len(nums) - 1
        while left <= right:
            center = (left + right) / 2
            

81 # Search in Rotated Sorted Array II

334 # Increasing Triplet Subsequence
class solution:
    def increasingTriplet(self,nums):
        a = b = None
        for n in nums:
            if a is None or a >= n:
                a = n
            elif b is None or b >= n:
                n = n
            else:
                return True
        return False

229 # Majority Element II
class solution:
    def majorityElement(self,nums):
        n1 = n2 = None
        c1 = c2 = 0
        for num in nums:
            if n1 == num:
                c1 += 1
            elif n2 == num:
                c2 += 1
            elif c1 == 0:
                n1,c1 == num,1
            elif c2 == 0:
                n2,c2 = num, 1
            else:
                c1,c2 = c1-1,c2-2
        size = len(nums)
        return [n for n in (n1,n2) if n is not None and nums.count(n)>size/3]

79 Word Search
62 Unique Paths
63 Unique Paths II
1 Two Sum
120 Triangle
78 Subsets
90 Subsets
34 Search for a Range
80 Remove Duplicates from Sorted Array II
238 Product of Array Except Self
class solution:
output[i]
    def 
31 Next Permutation

268 Missing Number
class solution:
    def missingNumber(self,nums):
        n = len(nums)
        return n*(n+1)/2 - sum(nums)

209 Minimum Size Subarray Sum
class solution:
    def 
64 Minimum Path Sum

53 Maximum Subarray
class solution:
# 单独开一个长度与nums相同的数组S，全部为0.
# 扫描S，如果S前数大于零，后数与前数相加求和
# 要么直接把nums[i]存到S[i]
    def maxSubArray(self,nums):
        S = [0 for i in xrange(len(nums))]
        S[0] = nums[0]
        for i in xrange(1,len(nums)):
            if S[i-1] > 0:
                S[i] = S[i-1] + nums[i]
            else:
                S[i] = nums[i]
        return max(S)

152 Maximum Product Subarray
# 建立变量mintmp,maxtmp,存储nums[i]与mintmp,maxtmp以及自身，判断大小，更新变量。
class solution:
    def maxProduct(self,nums):
        if len(nums) == 0:
            return 0
        mintmp = nums[0]
        maxtmp = nums[0]
        result = nums[0]
        for i in xrange(1,len(nums)):
            a = mintmp * nums[i]
            b = maxtmp * nums[i]
            c = nums[i]
            maxtmp = max(maxtmp,a)
            mintmp = min(mintmp,b)
            if maxtmp > result: result = maxtmp
            else: result
        return result

55 # Jump Game
基本思想应该是，在每个位置判断可以走得最远的位置，
以最远位置为基础，依次更新，直到达到指定位置为止。
class solution:
    def jump(self,nums):
        step = nums[0]
        for i in range(1,len(nums)):
            if step > 0:
                step -= 1 ##最少能走一步，看最左走几步
                step = max(step,nums[i])
            else:
                return False
        return True
289 Game of Life
162 Find Peak Element
11 Container With Most Water
105 COnstruct Binary Tree from Preorder and Inorder Traversal
106 COnstruct Binary Tree from Inorder and Postorder Traversal

39 # Combination Sum
class solution:
    def combinationSum(self,candidates,target):

        def DFS(candidates,target,start,valuelist):
            length = len(candidates)
            if target == 0 and valuelist not in res:
                return res.append(valuelist)
            for i in xrange(start,length):
                if target < candidates[i]:
                    return
                DFS(candidates,target-candidates[i],i,valuelist+[candidates[i]])
        res = []
        DFS(candidates,target,0,[])
        return res


216 Combination Sum III

40 # Combination Sum II
class solution:
    def combinationSum:

        def DFS(candidates,target,start,valuelist):
            if target == 0 and valuelist not in res:
                return res.append(valuelist)
            for i in xrange(start,length):
                if target < candidates[i]:
                    return 
                DFS(candidates,target-candidates[i],i+1,valuelist+[candidates[i]])
        res = []
        DFS(candidates,target,0,[])
        return res


121 Best Time to Buy and Sell Stock
122 Best Time to Buy and Sell Stock II

18 # 4SUm
class solution:
 需要用到哈希表的思路，这样可以空间换时间，以增加空间复杂度的代价来降低时间复杂度。
 首先建立一个字典dict，字典的key值为数组中每两个元素的和，
 每个key对应的value为这两个元素的下标组成的元组，元组不一定是唯一的。
 如对于num=[1,2,3,2]来说，dict={3:[(0,1),(0,3)], 4:[(0,2),(1,3)], 5:[(1,2),(2,3)]}。
 这样就可以检查target-key这个值在不在dict的key值中，
 如果target-key在dict中并且下标符合要求，那么就找到了这样的一组解。
 由于需要去重，这里选用set()类型的数据结构，即无序无重复元素集。
 最后将每个找出来的解(set()类型)转换成list类型输出即可。
    
    def fourSum(self,num,target):
        numLen = len(num)
        res = set()
        dict = {}
        if numLen < 4:
            return []
        num.sort()
        for p in range(numLen):
            for q in range(p+1,numLen):
                if num[p] + num[q] not in dict:
                    dict[num[p]+num[q]] = [(p,q)]
                else:
                    dict[num[p]+num[q]].append[(p,q)]

        for i in range(numLen):
            for j in range(i+1,numLen-2):
                T = target - num[i]-num[j]
                if T in dict:
                    for k in dict[T]:
                        if k[0] > j:
                            res.add(num[i],num[j],num[k[0]],num[k[1]])

        return [list(i) for i in res]

15 # 3SUm
class solution:
    def threeSum(self,num):
        res = []
        num=sorted(num)
        length = len(num)

        for i in xrange(0,length-2): # make sure a<b<c
            a = num[i]
            # remove duplicate a
            if i >= 1 and a == num[i-1]:
                continue
            j = i + 1
            k = length - 1
            while j < k:
                b = num[j]
                c = num[k]
                if b + c == -a:
                    res.append([a,b,c])
                    while j < k:
                        j += 1
                        k -= 1
                        if num[j] != b or num[k] != c:
                            break
                elif b + c > -a:
                    while j < k:
                        k -= 1
                        if num[k] != c:
                            break
                else: # b + c < -a 
                    while j < k:
                        j += 1
                        if num[j] != b:
                            break
            return res

16 # 3SUm Closest
使用一个变量mindiff来监测和与target之间的差值，如果差值为0，直接返回sum值。
class solution:
    def threeSumClosest(self,nums,target):
        ans = 0
        nums.sort()
        middiff = 100000
        for i in xrange(0,len(nums))):
            left = i+1
            right = len(nums) - 1
            while left < right:
                sum = nums[i] + nums[left]+nums[right]
                diff = abs(sum-target)
                if diff < middiff:
                    middiff = diff
                    ans =sum
                if sum == target:
                    return sum
                elif sum < target:
                    left += 1
                else: # sum > target
                    right -= 1
        return ans

228 # Summary Ranges
将逐一递增的整数序列化简为（起点->终点）即可。
class solution:
    def summaryRanges(self,nums):
        x = 0
        ans = []
        length = len(nums)
        while x < length:
            c = x 
            r = str(nums[x])
            while x + 1 < length and nums[x+1] == 1 + nums[x]:
                x += 1
            if x > c:
                r += '->' + str(nums[x])
            ans.append(r)
            x += 1
        return ans

66 # Plus One
class solution:
    def plusOne(self,digits):
        carry = 1
        i = len(digits) - 1
        while carry and i >= 0:
            d = digits[i]+ carry
            carry = d/10
            digits[i] = d% 10
        if carry:
            return [1] + digits
        else:
            return digits

217 # Contains Duplicate
class Solution:
    def containsDuplicate(self, nums):
        dict = {}
        for item in nums:
            if item in dict:
                return True
            else:
                dict[item] = 1
        return False

219 Contains Duplicate II








        



        