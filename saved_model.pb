╙╖8
╠в
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

·
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.1-0-g85c8b2a817f8Ьў-
Д
conv2d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_45/kernel
}
$conv2d_45/kernel/Read/ReadVariableOpReadVariableOpconv2d_45/kernel*&
_output_shapes
:*
dtype0
t
conv2d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_45/bias
m
"conv2d_45/bias/Read/ReadVariableOpReadVariableOpconv2d_45/bias*
_output_shapes
:*
dtype0
Д
conv2d_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_48/kernel
}
$conv2d_48/kernel/Read/ReadVariableOpReadVariableOpconv2d_48/kernel*&
_output_shapes
:*
dtype0
t
conv2d_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_48/bias
m
"conv2d_48/bias/Read/ReadVariableOpReadVariableOpconv2d_48/bias*
_output_shapes
:*
dtype0
Р
batch_normalization_60/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_60/gamma
Й
0batch_normalization_60/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_60/gamma*
_output_shapes
:*
dtype0
О
batch_normalization_60/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_60/beta
З
/batch_normalization_60/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_60/beta*
_output_shapes
:*
dtype0
Ь
"batch_normalization_60/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_60/moving_mean
Х
6batch_normalization_60/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_60/moving_mean*
_output_shapes
:*
dtype0
д
&batch_normalization_60/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_60/moving_variance
Э
:batch_normalization_60/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_60/moving_variance*
_output_shapes
:*
dtype0
Р
batch_normalization_64/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_64/gamma
Й
0batch_normalization_64/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_64/gamma*
_output_shapes
:*
dtype0
О
batch_normalization_64/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_64/beta
З
/batch_normalization_64/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_64/beta*
_output_shapes
:*
dtype0
Ь
"batch_normalization_64/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_64/moving_mean
Х
6batch_normalization_64/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_64/moving_mean*
_output_shapes
:*
dtype0
д
&batch_normalization_64/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_64/moving_variance
Э
:batch_normalization_64/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_64/moving_variance*
_output_shapes
:*
dtype0
Д
conv2d_46/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_46/kernel
}
$conv2d_46/kernel/Read/ReadVariableOpReadVariableOpconv2d_46/kernel*&
_output_shapes
: *
dtype0
t
conv2d_46/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_46/bias
m
"conv2d_46/bias/Read/ReadVariableOpReadVariableOpconv2d_46/bias*
_output_shapes
: *
dtype0
Д
conv2d_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_49/kernel
}
$conv2d_49/kernel/Read/ReadVariableOpReadVariableOpconv2d_49/kernel*&
_output_shapes
: *
dtype0
t
conv2d_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_49/bias
m
"conv2d_49/bias/Read/ReadVariableOpReadVariableOpconv2d_49/bias*
_output_shapes
: *
dtype0
Р
batch_normalization_61/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_61/gamma
Й
0batch_normalization_61/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_61/gamma*
_output_shapes
: *
dtype0
О
batch_normalization_61/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_61/beta
З
/batch_normalization_61/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_61/beta*
_output_shapes
: *
dtype0
Ь
"batch_normalization_61/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_61/moving_mean
Х
6batch_normalization_61/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_61/moving_mean*
_output_shapes
: *
dtype0
д
&batch_normalization_61/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_61/moving_variance
Э
:batch_normalization_61/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_61/moving_variance*
_output_shapes
: *
dtype0
Р
batch_normalization_65/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_65/gamma
Й
0batch_normalization_65/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_65/gamma*
_output_shapes
: *
dtype0
О
batch_normalization_65/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_65/beta
З
/batch_normalization_65/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_65/beta*
_output_shapes
: *
dtype0
Ь
"batch_normalization_65/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_65/moving_mean
Х
6batch_normalization_65/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_65/moving_mean*
_output_shapes
: *
dtype0
д
&batch_normalization_65/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_65/moving_variance
Э
:batch_normalization_65/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_65/moving_variance*
_output_shapes
: *
dtype0
Д
conv2d_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_47/kernel
}
$conv2d_47/kernel/Read/ReadVariableOpReadVariableOpconv2d_47/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_47/bias
m
"conv2d_47/bias/Read/ReadVariableOpReadVariableOpconv2d_47/bias*
_output_shapes
: *
dtype0
Д
conv2d_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_50/kernel
}
$conv2d_50/kernel/Read/ReadVariableOpReadVariableOpconv2d_50/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_50/bias
m
"conv2d_50/bias/Read/ReadVariableOpReadVariableOpconv2d_50/bias*
_output_shapes
: *
dtype0
Р
batch_normalization_62/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_62/gamma
Й
0batch_normalization_62/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_62/gamma*
_output_shapes
: *
dtype0
О
batch_normalization_62/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_62/beta
З
/batch_normalization_62/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_62/beta*
_output_shapes
: *
dtype0
Ь
"batch_normalization_62/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_62/moving_mean
Х
6batch_normalization_62/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_62/moving_mean*
_output_shapes
: *
dtype0
д
&batch_normalization_62/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_62/moving_variance
Э
:batch_normalization_62/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_62/moving_variance*
_output_shapes
: *
dtype0
Р
batch_normalization_66/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_66/gamma
Й
0batch_normalization_66/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_66/gamma*
_output_shapes
: *
dtype0
О
batch_normalization_66/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_66/beta
З
/batch_normalization_66/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_66/beta*
_output_shapes
: *
dtype0
Ь
"batch_normalization_66/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_66/moving_mean
Х
6batch_normalization_66/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_66/moving_mean*
_output_shapes
: *
dtype0
д
&batch_normalization_66/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_66/moving_variance
Э
:batch_normalization_66/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_66/moving_variance*
_output_shapes
: *
dtype0
|
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А* 
shared_namedense_30/kernel
u
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel* 
_output_shapes
:
А@А*
dtype0
s
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_30/bias
l
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes	
:А*
dtype0
|
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А* 
shared_namedense_32/kernel
u
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel* 
_output_shapes
:
А@А*
dtype0
s
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_32/bias
l
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes	
:А*
dtype0
С
batch_normalization_63/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_63/gamma
К
0batch_normalization_63/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_63/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_63/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_63/beta
И
/batch_normalization_63/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_63/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_63/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_63/moving_mean
Ц
6batch_normalization_63/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_63/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_63/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_63/moving_variance
Ю
:batch_normalization_63/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_63/moving_variance*
_output_shapes	
:А*
dtype0
С
batch_normalization_67/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_67/gamma
К
0batch_normalization_67/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_67/gamma*
_output_shapes	
:А*
dtype0
П
batch_normalization_67/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_67/beta
И
/batch_normalization_67/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_67/beta*
_output_shapes	
:А*
dtype0
Э
"batch_normalization_67/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"batch_normalization_67/moving_mean
Ц
6batch_normalization_67/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_67/moving_mean*
_output_shapes	
:А*
dtype0
е
&batch_normalization_67/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&batch_normalization_67/moving_variance
Ю
:batch_normalization_67/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_67/moving_variance*
_output_shapes	
:А*
dtype0
{
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_31/kernel
t
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes
:	А*
dtype0
r
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_31/bias
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes
:*
dtype0
{
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А* 
shared_namedense_33/kernel
t
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes
:	А*
dtype0
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
Т
Adam/conv2d_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_45/kernel/m
Л
+Adam/conv2d_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/m*&
_output_shapes
:*
dtype0
В
Adam/conv2d_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_45/bias/m
{
)Adam/conv2d_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_48/kernel/m
Л
+Adam/conv2d_48/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/kernel/m*&
_output_shapes
:*
dtype0
В
Adam/conv2d_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_48/bias/m
{
)Adam/conv2d_48/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/bias/m*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_60/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_60/gamma/m
Ч
7Adam/batch_normalization_60/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_60/gamma/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_60/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_60/beta/m
Х
6Adam/batch_normalization_60/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_60/beta/m*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_64/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_64/gamma/m
Ч
7Adam/batch_normalization_64/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_64/gamma/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_64/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_64/beta/m
Х
6Adam/batch_normalization_64/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_64/beta/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_46/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_46/kernel/m
Л
+Adam/conv2d_46/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/m*&
_output_shapes
: *
dtype0
В
Adam/conv2d_46/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_46/bias/m
{
)Adam/conv2d_46/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/m*
_output_shapes
: *
dtype0
Т
Adam/conv2d_49/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_49/kernel/m
Л
+Adam/conv2d_49/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/kernel/m*&
_output_shapes
: *
dtype0
В
Adam/conv2d_49/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_49/bias/m
{
)Adam/conv2d_49/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/bias/m*
_output_shapes
: *
dtype0
Ю
#Adam/batch_normalization_61/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_61/gamma/m
Ч
7Adam/batch_normalization_61/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_61/gamma/m*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_61/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_61/beta/m
Х
6Adam/batch_normalization_61/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_61/beta/m*
_output_shapes
: *
dtype0
Ю
#Adam/batch_normalization_65/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_65/gamma/m
Ч
7Adam/batch_normalization_65/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_65/gamma/m*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_65/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_65/beta/m
Х
6Adam/batch_normalization_65/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_65/beta/m*
_output_shapes
: *
dtype0
Т
Adam/conv2d_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_47/kernel/m
Л
+Adam/conv2d_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/m*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_47/bias/m
{
)Adam/conv2d_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/m*
_output_shapes
: *
dtype0
Т
Adam/conv2d_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_50/kernel/m
Л
+Adam/conv2d_50/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/kernel/m*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_50/bias/m
{
)Adam/conv2d_50/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/bias/m*
_output_shapes
: *
dtype0
Ю
#Adam/batch_normalization_62/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_62/gamma/m
Ч
7Adam/batch_normalization_62/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_62/gamma/m*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_62/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_62/beta/m
Х
6Adam/batch_normalization_62/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_62/beta/m*
_output_shapes
: *
dtype0
Ю
#Adam/batch_normalization_66/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_66/gamma/m
Ч
7Adam/batch_normalization_66/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_66/gamma/m*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_66/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_66/beta/m
Х
6Adam/batch_normalization_66/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_66/beta/m*
_output_shapes
: *
dtype0
К
Adam/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*'
shared_nameAdam/dense_30/kernel/m
Г
*Adam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/m* 
_output_shapes
:
А@А*
dtype0
Б
Adam/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_30/bias/m
z
(Adam/dense_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/m*
_output_shapes	
:А*
dtype0
К
Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*'
shared_nameAdam/dense_32/kernel/m
Г
*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m* 
_output_shapes
:
А@А*
dtype0
Б
Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_32/bias/m
z
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_63/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_63/gamma/m
Ш
7Adam/batch_normalization_63/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_63/gamma/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_63/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_63/beta/m
Ц
6Adam/batch_normalization_63/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_63/beta/m*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_67/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_67/gamma/m
Ш
7Adam/batch_normalization_67/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_67/gamma/m*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_67/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_67/beta/m
Ц
6Adam/batch_normalization_67/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_67/beta/m*
_output_shapes	
:А*
dtype0
Й
Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_31/kernel/m
В
*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m*
_output_shapes
:	А*
dtype0
А
Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/m
y
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes
:*
dtype0
Й
Adam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_33/kernel/m
В
*Adam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/m*
_output_shapes
:	А*
dtype0
А
Adam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/m
y
(Adam/dense_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_45/kernel/v
Л
+Adam/conv2d_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/v*&
_output_shapes
:*
dtype0
В
Adam/conv2d_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_45/bias/v
{
)Adam/conv2d_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/v*
_output_shapes
:*
dtype0
Т
Adam/conv2d_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_48/kernel/v
Л
+Adam/conv2d_48/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/kernel/v*&
_output_shapes
:*
dtype0
В
Adam/conv2d_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_48/bias/v
{
)Adam/conv2d_48/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/bias/v*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_60/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_60/gamma/v
Ч
7Adam/batch_normalization_60/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_60/gamma/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_60/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_60/beta/v
Х
6Adam/batch_normalization_60/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_60/beta/v*
_output_shapes
:*
dtype0
Ю
#Adam/batch_normalization_64/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_64/gamma/v
Ч
7Adam/batch_normalization_64/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_64/gamma/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_64/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_64/beta/v
Х
6Adam/batch_normalization_64/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_64/beta/v*
_output_shapes
:*
dtype0
Т
Adam/conv2d_46/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_46/kernel/v
Л
+Adam/conv2d_46/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/kernel/v*&
_output_shapes
: *
dtype0
В
Adam/conv2d_46/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_46/bias/v
{
)Adam/conv2d_46/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_46/bias/v*
_output_shapes
: *
dtype0
Т
Adam/conv2d_49/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_49/kernel/v
Л
+Adam/conv2d_49/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/kernel/v*&
_output_shapes
: *
dtype0
В
Adam/conv2d_49/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_49/bias/v
{
)Adam/conv2d_49/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/bias/v*
_output_shapes
: *
dtype0
Ю
#Adam/batch_normalization_61/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_61/gamma/v
Ч
7Adam/batch_normalization_61/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_61/gamma/v*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_61/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_61/beta/v
Х
6Adam/batch_normalization_61/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_61/beta/v*
_output_shapes
: *
dtype0
Ю
#Adam/batch_normalization_65/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_65/gamma/v
Ч
7Adam/batch_normalization_65/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_65/gamma/v*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_65/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_65/beta/v
Х
6Adam/batch_normalization_65/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_65/beta/v*
_output_shapes
: *
dtype0
Т
Adam/conv2d_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_47/kernel/v
Л
+Adam/conv2d_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/v*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_47/bias/v
{
)Adam/conv2d_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/v*
_output_shapes
: *
dtype0
Т
Adam/conv2d_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_50/kernel/v
Л
+Adam/conv2d_50/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/kernel/v*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_50/bias/v
{
)Adam/conv2d_50/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/bias/v*
_output_shapes
: *
dtype0
Ю
#Adam/batch_normalization_62/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_62/gamma/v
Ч
7Adam/batch_normalization_62/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_62/gamma/v*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_62/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_62/beta/v
Х
6Adam/batch_normalization_62/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_62/beta/v*
_output_shapes
: *
dtype0
Ю
#Adam/batch_normalization_66/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_66/gamma/v
Ч
7Adam/batch_normalization_66/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_66/gamma/v*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_66/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_66/beta/v
Х
6Adam/batch_normalization_66/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_66/beta/v*
_output_shapes
: *
dtype0
К
Adam/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*'
shared_nameAdam/dense_30/kernel/v
Г
*Adam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/v* 
_output_shapes
:
А@А*
dtype0
Б
Adam/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_30/bias/v
z
(Adam/dense_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/v*
_output_shapes	
:А*
dtype0
К
Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*'
shared_nameAdam/dense_32/kernel/v
Г
*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v* 
_output_shapes
:
А@А*
dtype0
Б
Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdam/dense_32/bias/v
z
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_63/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_63/gamma/v
Ш
7Adam/batch_normalization_63/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_63/gamma/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_63/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_63/beta/v
Ц
6Adam/batch_normalization_63/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_63/beta/v*
_output_shapes	
:А*
dtype0
Я
#Adam/batch_normalization_67/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#Adam/batch_normalization_67/gamma/v
Ш
7Adam/batch_normalization_67/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_67/gamma/v*
_output_shapes	
:А*
dtype0
Э
"Adam/batch_normalization_67/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*3
shared_name$"Adam/batch_normalization_67/beta/v
Ц
6Adam/batch_normalization_67/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_67/beta/v*
_output_shapes	
:А*
dtype0
Й
Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_31/kernel/v
В
*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v*
_output_shapes
:	А*
dtype0
А
Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/v
y
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes
:*
dtype0
Й
Adam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*'
shared_nameAdam/dense_33/kernel/v
В
*Adam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/v*
_output_shapes
:	А*
dtype0
А
Adam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/v
y
(Adam/dense_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
╔О
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ГО
value°НBЇН BьН
├	
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer-19
layer-20
layer_with_weights-8
layer-21
layer_with_weights-9
layer-22
layer-23
layer-24
layer_with_weights-10
layer-25
layer_with_weights-11
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer_with_weights-12
"layer-33
#layer_with_weights-13
#layer-34
$layer-35
%layer-36
&layer_with_weights-14
&layer-37
'layer_with_weights-15
'layer-38
(layer-39
)layer-40
*layer_with_weights-16
*layer-41
+layer_with_weights-17
+layer-42
,layer-43
-layer-44
.	optimizer
/loss
0regularization_losses
1trainable_variables
2	variables
3	keras_api
4
signatures
 
h

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
R
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
Ч
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
Ч
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
R
[regularization_losses
\trainable_variables
]	variables
^	keras_api
R
_regularization_losses
`trainable_variables
a	variables
b	keras_api
R
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
R
gregularization_losses
htrainable_variables
i	variables
j	keras_api
h

kkernel
lbias
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
h

qkernel
rbias
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
R
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
R
{regularization_losses
|trainable_variables
}	variables
~	keras_api
Я
axis

Аgamma
	Бbeta
Вmoving_mean
Гmoving_variance
Дregularization_losses
Еtrainable_variables
Ж	variables
З	keras_api
а
	Иaxis

Йgamma
	Кbeta
Лmoving_mean
Мmoving_variance
Нregularization_losses
Оtrainable_variables
П	variables
Р	keras_api
V
Сregularization_losses
Тtrainable_variables
У	variables
Ф	keras_api
V
Хregularization_losses
Цtrainable_variables
Ч	variables
Ш	keras_api
V
Щregularization_losses
Ъtrainable_variables
Ы	variables
Ь	keras_api
V
Эregularization_losses
Юtrainable_variables
Я	variables
а	keras_api
n
бkernel
	вbias
гregularization_losses
дtrainable_variables
е	variables
ж	keras_api
n
зkernel
	иbias
йregularization_losses
кtrainable_variables
л	variables
м	keras_api
V
нregularization_losses
оtrainable_variables
п	variables
░	keras_api
V
▒regularization_losses
▓trainable_variables
│	variables
┤	keras_api
а
	╡axis

╢gamma
	╖beta
╕moving_mean
╣moving_variance
║regularization_losses
╗trainable_variables
╝	variables
╜	keras_api
а
	╛axis

┐gamma
	└beta
┴moving_mean
┬moving_variance
├regularization_losses
─trainable_variables
┼	variables
╞	keras_api
V
╟regularization_losses
╚trainable_variables
╔	variables
╩	keras_api
V
╦regularization_losses
╠trainable_variables
═	variables
╬	keras_api
V
╧regularization_losses
╨trainable_variables
╤	variables
╥	keras_api
V
╙regularization_losses
╘trainable_variables
╒	variables
╓	keras_api
V
╫regularization_losses
╪trainable_variables
┘	variables
┌	keras_api
V
█regularization_losses
▄trainable_variables
▌	variables
▐	keras_api
n
▀kernel
	рbias
сregularization_losses
тtrainable_variables
у	variables
ф	keras_api
n
хkernel
	цbias
чregularization_losses
шtrainable_variables
щ	variables
ъ	keras_api
V
ыregularization_losses
ьtrainable_variables
э	variables
ю	keras_api
V
яregularization_losses
Ёtrainable_variables
ё	variables
Є	keras_api
а
	єaxis

Їgamma
	їbeta
Ўmoving_mean
ўmoving_variance
°regularization_losses
∙trainable_variables
·	variables
√	keras_api
а
	№axis

¤gamma
	■beta
 moving_mean
Аmoving_variance
Бregularization_losses
Вtrainable_variables
Г	variables
Д	keras_api
V
Еregularization_losses
Жtrainable_variables
З	variables
И	keras_api
V
Йregularization_losses
Кtrainable_variables
Л	variables
М	keras_api
n
Нkernel
	Оbias
Пregularization_losses
Рtrainable_variables
С	variables
Т	keras_api
n
Уkernel
	Фbias
Хregularization_losses
Цtrainable_variables
Ч	variables
Ш	keras_api
V
Щregularization_losses
Ъtrainable_variables
Ы	variables
Ь	keras_api
V
Эregularization_losses
Юtrainable_variables
Я	variables
а	keras_api
┼
	бiter
вbeta_1
гbeta_2

дdecay
еlearning_rate5mв6mг;mд<mеJmжKmзSmиTmйkmкlmлqmмrmн	Аmо	Бmп	Йm░	Кm▒	бm▓	вm│	зm┤	иm╡	╢m╢	╖m╖	┐m╕	└m╣	▀m║	рm╗	хm╝	цm╜	Їm╛	їm┐	¤m└	■m┴	Нm┬	Оm├	Уm─	Фm┼5v╞6v╟;v╚<v╔Jv╩Kv╦Sv╠Tv═kv╬lv╧qv╨rv╤	Аv╥	Бv╙	Йv╘	Кv╒	бv╓	вv╫	зv╪	иv┘	╢v┌	╖v█	┐v▄	└v▌	▀v▐	рv▀	хvр	цvс	Їvт	їvу	¤vф	■vх	Нvц	Оvч	Уvш	Фvщ
 
 
о
50
61
;2
<3
J4
K5
S6
T7
k8
l9
q10
r11
А12
Б13
Й14
К15
б16
в17
з18
и19
╢20
╖21
┐22
└23
▀24
р25
х26
ц27
Ї28
ї29
¤30
■31
Н32
О33
У34
Ф35
║
50
61
;2
<3
J4
K5
L6
M7
S8
T9
U10
V11
k12
l13
q14
r15
А16
Б17
В18
Г19
Й20
К21
Л22
М23
б24
в25
з26
и27
╢28
╖29
╕30
╣31
┐32
└33
┴34
┬35
▀36
р37
х38
ц39
Ї40
ї41
Ў42
ў43
¤44
■45
 46
А47
Н48
О49
У50
Ф51
▓
жlayer_metrics
0regularization_losses
 зlayer_regularization_losses
1trainable_variables
иmetrics
2	variables
йlayers
кnon_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_45/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_45/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
▓
лlayers
мlayer_metrics
 нlayer_regularization_losses
7regularization_losses
8trainable_variables
9	variables
оmetrics
пnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_48/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_48/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
▓
░layers
▒layer_metrics
 ▓layer_regularization_losses
=regularization_losses
>trainable_variables
?	variables
│metrics
┤non_trainable_variables
 
 
 
▓
╡layers
╢layer_metrics
 ╖layer_regularization_losses
Aregularization_losses
Btrainable_variables
C	variables
╕metrics
╣non_trainable_variables
 
 
 
▓
║layers
╗layer_metrics
 ╝layer_regularization_losses
Eregularization_losses
Ftrainable_variables
G	variables
╜metrics
╛non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_60/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_60/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_60/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_60/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
L2
M3
▓
┐layers
└layer_metrics
 ┴layer_regularization_losses
Nregularization_losses
Otrainable_variables
P	variables
┬metrics
├non_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_64/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_64/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_64/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_64/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

S0
T1
U2
V3
▓
─layers
┼layer_metrics
 ╞layer_regularization_losses
Wregularization_losses
Xtrainable_variables
Y	variables
╟metrics
╚non_trainable_variables
 
 
 
▓
╔layers
╩layer_metrics
 ╦layer_regularization_losses
[regularization_losses
\trainable_variables
]	variables
╠metrics
═non_trainable_variables
 
 
 
▓
╬layers
╧layer_metrics
 ╨layer_regularization_losses
_regularization_losses
`trainable_variables
a	variables
╤metrics
╥non_trainable_variables
 
 
 
▓
╙layers
╘layer_metrics
 ╒layer_regularization_losses
cregularization_losses
dtrainable_variables
e	variables
╓metrics
╫non_trainable_variables
 
 
 
▓
╪layers
┘layer_metrics
 ┌layer_regularization_losses
gregularization_losses
htrainable_variables
i	variables
█metrics
▄non_trainable_variables
\Z
VARIABLE_VALUEconv2d_46/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_46/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

k0
l1

k0
l1
▓
▌layers
▐layer_metrics
 ▀layer_regularization_losses
mregularization_losses
ntrainable_variables
o	variables
рmetrics
сnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_49/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_49/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

q0
r1
▓
тlayers
уlayer_metrics
 фlayer_regularization_losses
sregularization_losses
ttrainable_variables
u	variables
хmetrics
цnon_trainable_variables
 
 
 
▓
чlayers
шlayer_metrics
 щlayer_regularization_losses
wregularization_losses
xtrainable_variables
y	variables
ъmetrics
ыnon_trainable_variables
 
 
 
▓
ьlayers
эlayer_metrics
 юlayer_regularization_losses
{regularization_losses
|trainable_variables
}	variables
яmetrics
Ёnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_61/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_61/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_61/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_61/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

А0
Б1
 
А0
Б1
В2
Г3
╡
ёlayers
Єlayer_metrics
 єlayer_regularization_losses
Дregularization_losses
Еtrainable_variables
Ж	variables
Їmetrics
їnon_trainable_variables
 
ge
VARIABLE_VALUEbatch_normalization_65/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_65/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_65/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_65/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

Й0
К1
 
Й0
К1
Л2
М3
╡
Ўlayers
ўlayer_metrics
 °layer_regularization_losses
Нregularization_losses
Оtrainable_variables
П	variables
∙metrics
·non_trainable_variables
 
 
 
╡
√layers
№layer_metrics
 ¤layer_regularization_losses
Сregularization_losses
Тtrainable_variables
У	variables
■metrics
 non_trainable_variables
 
 
 
╡
Аlayers
Бlayer_metrics
 Вlayer_regularization_losses
Хregularization_losses
Цtrainable_variables
Ч	variables
Гmetrics
Дnon_trainable_variables
 
 
 
╡
Еlayers
Жlayer_metrics
 Зlayer_regularization_losses
Щregularization_losses
Ъtrainable_variables
Ы	variables
Иmetrics
Йnon_trainable_variables
 
 
 
╡
Кlayers
Лlayer_metrics
 Мlayer_regularization_losses
Эregularization_losses
Юtrainable_variables
Я	variables
Нmetrics
Оnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_47/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_47/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

б0
в1

б0
в1
╡
Пlayers
Рlayer_metrics
 Сlayer_regularization_losses
гregularization_losses
дtrainable_variables
е	variables
Тmetrics
Уnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_50/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_50/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

з0
и1

з0
и1
╡
Фlayers
Хlayer_metrics
 Цlayer_regularization_losses
йregularization_losses
кtrainable_variables
л	variables
Чmetrics
Шnon_trainable_variables
 
 
 
╡
Щlayers
Ъlayer_metrics
 Ыlayer_regularization_losses
нregularization_losses
оtrainable_variables
п	variables
Ьmetrics
Эnon_trainable_variables
 
 
 
╡
Юlayers
Яlayer_metrics
 аlayer_regularization_losses
▒regularization_losses
▓trainable_variables
│	variables
бmetrics
вnon_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_62/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_62/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_62/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_62/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

╢0
╖1
 
╢0
╖1
╕2
╣3
╡
гlayers
дlayer_metrics
 еlayer_regularization_losses
║regularization_losses
╗trainable_variables
╝	variables
жmetrics
зnon_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_66/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_66/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_66/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_66/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

┐0
└1
 
┐0
└1
┴2
┬3
╡
иlayers
йlayer_metrics
 кlayer_regularization_losses
├regularization_losses
─trainable_variables
┼	variables
лmetrics
мnon_trainable_variables
 
 
 
╡
нlayers
оlayer_metrics
 пlayer_regularization_losses
╟regularization_losses
╚trainable_variables
╔	variables
░metrics
▒non_trainable_variables
 
 
 
╡
▓layers
│layer_metrics
 ┤layer_regularization_losses
╦regularization_losses
╠trainable_variables
═	variables
╡metrics
╢non_trainable_variables
 
 
 
╡
╖layers
╕layer_metrics
 ╣layer_regularization_losses
╧regularization_losses
╨trainable_variables
╤	variables
║metrics
╗non_trainable_variables
 
 
 
╡
╝layers
╜layer_metrics
 ╛layer_regularization_losses
╙regularization_losses
╘trainable_variables
╒	variables
┐metrics
└non_trainable_variables
 
 
 
╡
┴layers
┬layer_metrics
 ├layer_regularization_losses
╫regularization_losses
╪trainable_variables
┘	variables
─metrics
┼non_trainable_variables
 
 
 
╡
╞layers
╟layer_metrics
 ╚layer_regularization_losses
█regularization_losses
▄trainable_variables
▌	variables
╔metrics
╩non_trainable_variables
\Z
VARIABLE_VALUEdense_30/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_30/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

▀0
р1

▀0
р1
╡
╦layers
╠layer_metrics
 ═layer_regularization_losses
сregularization_losses
тtrainable_variables
у	variables
╬metrics
╧non_trainable_variables
\Z
VARIABLE_VALUEdense_32/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_32/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

х0
ц1

х0
ц1
╡
╨layers
╤layer_metrics
 ╥layer_regularization_losses
чregularization_losses
шtrainable_variables
щ	variables
╙metrics
╘non_trainable_variables
 
 
 
╡
╒layers
╓layer_metrics
 ╫layer_regularization_losses
ыregularization_losses
ьtrainable_variables
э	variables
╪metrics
┘non_trainable_variables
 
 
 
╡
┌layers
█layer_metrics
 ▄layer_regularization_losses
яregularization_losses
Ёtrainable_variables
ё	variables
▌metrics
▐non_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_63/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_63/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_63/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_63/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

Ї0
ї1
 
Ї0
ї1
Ў2
ў3
╡
▀layers
рlayer_metrics
 сlayer_regularization_losses
°regularization_losses
∙trainable_variables
·	variables
тmetrics
уnon_trainable_variables
 
hf
VARIABLE_VALUEbatch_normalization_67/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_67/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE"batch_normalization_67/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE&batch_normalization_67/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

¤0
■1
 
¤0
■1
 2
А3
╡
фlayers
хlayer_metrics
 цlayer_regularization_losses
Бregularization_losses
Вtrainable_variables
Г	variables
чmetrics
шnon_trainable_variables
 
 
 
╡
щlayers
ъlayer_metrics
 ыlayer_regularization_losses
Еregularization_losses
Жtrainable_variables
З	variables
ьmetrics
эnon_trainable_variables
 
 
 
╡
юlayers
яlayer_metrics
 Ёlayer_regularization_losses
Йregularization_losses
Кtrainable_variables
Л	variables
ёmetrics
Єnon_trainable_variables
\Z
VARIABLE_VALUEdense_31/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_31/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Н0
О1

Н0
О1
╡
єlayers
Їlayer_metrics
 їlayer_regularization_losses
Пregularization_losses
Рtrainable_variables
С	variables
Ўmetrics
ўnon_trainable_variables
\Z
VARIABLE_VALUEdense_33/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_33/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

У0
Ф1

У0
Ф1
╡
°layers
∙layer_metrics
 ·layer_regularization_losses
Хregularization_losses
Цtrainable_variables
Ч	variables
√metrics
№non_trainable_variables
 
 
 
╡
¤layers
■layer_metrics
  layer_regularization_losses
Щregularization_losses
Ъtrainable_variables
Ы	variables
Аmetrics
Бnon_trainable_variables
 
 
 
╡
Вlayers
Гlayer_metrics
 Дlayer_regularization_losses
Эregularization_losses
Юtrainable_variables
Я	variables
Еmetrics
Жnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
(
З0
И1
Й2
К3
Л4
▐
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
В
L0
M1
U2
V3
В4
Г5
Л6
М7
╕8
╣9
┴10
┬11
Ў12
ў13
 14
А15
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

L0
M1
 
 
 
 

U0
V1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

В0
Г1
 
 
 
 

Л0
М1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

╕0
╣1
 
 
 
 

┴0
┬1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Ў0
ў1
 
 
 
 

 0
А1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Мtotal

Нcount
О	variables
П	keras_api
8

Рtotal

Сcount
Т	variables
У	keras_api
8

Фtotal

Хcount
Ц	variables
Ч	keras_api
I

Шtotal

Щcount
Ъ
_fn_kwargs
Ы	variables
Ь	keras_api
I

Эtotal

Юcount
Я
_fn_kwargs
а	variables
б	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

М0
Н1

О	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

Р0
С1

Т	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

Ф0
Х1

Ц	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ш0
Щ1

Ы	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE
 

Э0
Ю1

а	variables
}
VARIABLE_VALUEAdam/conv2d_45/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_45/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_48/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_48/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_60/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_60/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_64/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_64/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_46/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_46/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_49/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_49/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_61/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_61/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_65/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_65/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_47/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_47/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_50/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_50/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_62/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_62/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_66/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_66/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_30/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_30/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_32/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_32/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_63/gamma/mRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_63/beta/mQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_67/gamma/mRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_67/beta/mQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_31/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_31/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_33/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_33/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_45/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_45/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_48/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_48/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_60/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_60/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_64/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_64/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_46/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_46/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_49/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_49/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_61/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_61/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#Adam/batch_normalization_65/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"Adam/batch_normalization_65/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_47/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_47/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_50/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_50/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_62/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_62/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_66/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_66/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_30/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_30/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_32/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_32/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_63/gamma/vRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_63/beta/vQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#Adam/batch_normalization_67/gamma/vRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_67/beta/vQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_31/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_31/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_33/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_33/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
О
serving_default_input_6Placeholder*1
_output_shapes
:         ╞╞*
dtype0*&
shape:         ╞╞
┤
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6conv2d_48/kernelconv2d_48/biasconv2d_45/kernelconv2d_45/biasbatch_normalization_64/gammabatch_normalization_64/beta"batch_normalization_64/moving_mean&batch_normalization_64/moving_variancebatch_normalization_60/gammabatch_normalization_60/beta"batch_normalization_60/moving_mean&batch_normalization_60/moving_varianceconv2d_49/kernelconv2d_49/biasconv2d_46/kernelconv2d_46/biasbatch_normalization_65/gammabatch_normalization_65/beta"batch_normalization_65/moving_mean&batch_normalization_65/moving_variancebatch_normalization_61/gammabatch_normalization_61/beta"batch_normalization_61/moving_mean&batch_normalization_61/moving_varianceconv2d_50/kernelconv2d_50/biasconv2d_47/kernelconv2d_47/biasbatch_normalization_66/gammabatch_normalization_66/beta"batch_normalization_66/moving_mean&batch_normalization_66/moving_variancebatch_normalization_62/gammabatch_normalization_62/beta"batch_normalization_62/moving_mean&batch_normalization_62/moving_variancedense_32/kerneldense_32/biasdense_30/kerneldense_30/bias&batch_normalization_67/moving_variancebatch_normalization_67/gamma"batch_normalization_67/moving_meanbatch_normalization_67/beta&batch_normalization_63/moving_variancebatch_normalization_63/gamma"batch_normalization_63/moving_meanbatch_normalization_63/betadense_33/kerneldense_33/biasdense_31/kerneldense_31/bias*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_40794
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ч5
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_45/kernel/Read/ReadVariableOp"conv2d_45/bias/Read/ReadVariableOp$conv2d_48/kernel/Read/ReadVariableOp"conv2d_48/bias/Read/ReadVariableOp0batch_normalization_60/gamma/Read/ReadVariableOp/batch_normalization_60/beta/Read/ReadVariableOp6batch_normalization_60/moving_mean/Read/ReadVariableOp:batch_normalization_60/moving_variance/Read/ReadVariableOp0batch_normalization_64/gamma/Read/ReadVariableOp/batch_normalization_64/beta/Read/ReadVariableOp6batch_normalization_64/moving_mean/Read/ReadVariableOp:batch_normalization_64/moving_variance/Read/ReadVariableOp$conv2d_46/kernel/Read/ReadVariableOp"conv2d_46/bias/Read/ReadVariableOp$conv2d_49/kernel/Read/ReadVariableOp"conv2d_49/bias/Read/ReadVariableOp0batch_normalization_61/gamma/Read/ReadVariableOp/batch_normalization_61/beta/Read/ReadVariableOp6batch_normalization_61/moving_mean/Read/ReadVariableOp:batch_normalization_61/moving_variance/Read/ReadVariableOp0batch_normalization_65/gamma/Read/ReadVariableOp/batch_normalization_65/beta/Read/ReadVariableOp6batch_normalization_65/moving_mean/Read/ReadVariableOp:batch_normalization_65/moving_variance/Read/ReadVariableOp$conv2d_47/kernel/Read/ReadVariableOp"conv2d_47/bias/Read/ReadVariableOp$conv2d_50/kernel/Read/ReadVariableOp"conv2d_50/bias/Read/ReadVariableOp0batch_normalization_62/gamma/Read/ReadVariableOp/batch_normalization_62/beta/Read/ReadVariableOp6batch_normalization_62/moving_mean/Read/ReadVariableOp:batch_normalization_62/moving_variance/Read/ReadVariableOp0batch_normalization_66/gamma/Read/ReadVariableOp/batch_normalization_66/beta/Read/ReadVariableOp6batch_normalization_66/moving_mean/Read/ReadVariableOp:batch_normalization_66/moving_variance/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp0batch_normalization_63/gamma/Read/ReadVariableOp/batch_normalization_63/beta/Read/ReadVariableOp6batch_normalization_63/moving_mean/Read/ReadVariableOp:batch_normalization_63/moving_variance/Read/ReadVariableOp0batch_normalization_67/gamma/Read/ReadVariableOp/batch_normalization_67/beta/Read/ReadVariableOp6batch_normalization_67/moving_mean/Read/ReadVariableOp:batch_normalization_67/moving_variance/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOp+Adam/conv2d_45/kernel/m/Read/ReadVariableOp)Adam/conv2d_45/bias/m/Read/ReadVariableOp+Adam/conv2d_48/kernel/m/Read/ReadVariableOp)Adam/conv2d_48/bias/m/Read/ReadVariableOp7Adam/batch_normalization_60/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_60/beta/m/Read/ReadVariableOp7Adam/batch_normalization_64/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_64/beta/m/Read/ReadVariableOp+Adam/conv2d_46/kernel/m/Read/ReadVariableOp)Adam/conv2d_46/bias/m/Read/ReadVariableOp+Adam/conv2d_49/kernel/m/Read/ReadVariableOp)Adam/conv2d_49/bias/m/Read/ReadVariableOp7Adam/batch_normalization_61/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_61/beta/m/Read/ReadVariableOp7Adam/batch_normalization_65/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_65/beta/m/Read/ReadVariableOp+Adam/conv2d_47/kernel/m/Read/ReadVariableOp)Adam/conv2d_47/bias/m/Read/ReadVariableOp+Adam/conv2d_50/kernel/m/Read/ReadVariableOp)Adam/conv2d_50/bias/m/Read/ReadVariableOp7Adam/batch_normalization_62/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_62/beta/m/Read/ReadVariableOp7Adam/batch_normalization_66/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_66/beta/m/Read/ReadVariableOp*Adam/dense_30/kernel/m/Read/ReadVariableOp(Adam/dense_30/bias/m/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp7Adam/batch_normalization_63/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_63/beta/m/Read/ReadVariableOp7Adam/batch_normalization_67/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_67/beta/m/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp*Adam/dense_33/kernel/m/Read/ReadVariableOp(Adam/dense_33/bias/m/Read/ReadVariableOp+Adam/conv2d_45/kernel/v/Read/ReadVariableOp)Adam/conv2d_45/bias/v/Read/ReadVariableOp+Adam/conv2d_48/kernel/v/Read/ReadVariableOp)Adam/conv2d_48/bias/v/Read/ReadVariableOp7Adam/batch_normalization_60/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_60/beta/v/Read/ReadVariableOp7Adam/batch_normalization_64/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_64/beta/v/Read/ReadVariableOp+Adam/conv2d_46/kernel/v/Read/ReadVariableOp)Adam/conv2d_46/bias/v/Read/ReadVariableOp+Adam/conv2d_49/kernel/v/Read/ReadVariableOp)Adam/conv2d_49/bias/v/Read/ReadVariableOp7Adam/batch_normalization_61/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_61/beta/v/Read/ReadVariableOp7Adam/batch_normalization_65/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_65/beta/v/Read/ReadVariableOp+Adam/conv2d_47/kernel/v/Read/ReadVariableOp)Adam/conv2d_47/bias/v/Read/ReadVariableOp+Adam/conv2d_50/kernel/v/Read/ReadVariableOp)Adam/conv2d_50/bias/v/Read/ReadVariableOp7Adam/batch_normalization_62/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_62/beta/v/Read/ReadVariableOp7Adam/batch_normalization_66/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_66/beta/v/Read/ReadVariableOp*Adam/dense_30/kernel/v/Read/ReadVariableOp(Adam/dense_30/bias/v/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp7Adam/batch_normalization_63/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_63/beta/v/Read/ReadVariableOp7Adam/batch_normalization_67/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_67/beta/v/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOp*Adam/dense_33/kernel/v/Read/ReadVariableOp(Adam/dense_33/bias/v/Read/ReadVariableOpConst*Ы
TinУ
Р2Н	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *'
f"R 
__inference__traced_save_43432
Ж 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_45/kernelconv2d_45/biasconv2d_48/kernelconv2d_48/biasbatch_normalization_60/gammabatch_normalization_60/beta"batch_normalization_60/moving_mean&batch_normalization_60/moving_variancebatch_normalization_64/gammabatch_normalization_64/beta"batch_normalization_64/moving_mean&batch_normalization_64/moving_varianceconv2d_46/kernelconv2d_46/biasconv2d_49/kernelconv2d_49/biasbatch_normalization_61/gammabatch_normalization_61/beta"batch_normalization_61/moving_mean&batch_normalization_61/moving_variancebatch_normalization_65/gammabatch_normalization_65/beta"batch_normalization_65/moving_mean&batch_normalization_65/moving_varianceconv2d_47/kernelconv2d_47/biasconv2d_50/kernelconv2d_50/biasbatch_normalization_62/gammabatch_normalization_62/beta"batch_normalization_62/moving_mean&batch_normalization_62/moving_variancebatch_normalization_66/gammabatch_normalization_66/beta"batch_normalization_66/moving_mean&batch_normalization_66/moving_variancedense_30/kerneldense_30/biasdense_32/kerneldense_32/biasbatch_normalization_63/gammabatch_normalization_63/beta"batch_normalization_63/moving_mean&batch_normalization_63/moving_variancebatch_normalization_67/gammabatch_normalization_67/beta"batch_normalization_67/moving_mean&batch_normalization_67/moving_variancedense_31/kerneldense_31/biasdense_33/kerneldense_33/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4Adam/conv2d_45/kernel/mAdam/conv2d_45/bias/mAdam/conv2d_48/kernel/mAdam/conv2d_48/bias/m#Adam/batch_normalization_60/gamma/m"Adam/batch_normalization_60/beta/m#Adam/batch_normalization_64/gamma/m"Adam/batch_normalization_64/beta/mAdam/conv2d_46/kernel/mAdam/conv2d_46/bias/mAdam/conv2d_49/kernel/mAdam/conv2d_49/bias/m#Adam/batch_normalization_61/gamma/m"Adam/batch_normalization_61/beta/m#Adam/batch_normalization_65/gamma/m"Adam/batch_normalization_65/beta/mAdam/conv2d_47/kernel/mAdam/conv2d_47/bias/mAdam/conv2d_50/kernel/mAdam/conv2d_50/bias/m#Adam/batch_normalization_62/gamma/m"Adam/batch_normalization_62/beta/m#Adam/batch_normalization_66/gamma/m"Adam/batch_normalization_66/beta/mAdam/dense_30/kernel/mAdam/dense_30/bias/mAdam/dense_32/kernel/mAdam/dense_32/bias/m#Adam/batch_normalization_63/gamma/m"Adam/batch_normalization_63/beta/m#Adam/batch_normalization_67/gamma/m"Adam/batch_normalization_67/beta/mAdam/dense_31/kernel/mAdam/dense_31/bias/mAdam/dense_33/kernel/mAdam/dense_33/bias/mAdam/conv2d_45/kernel/vAdam/conv2d_45/bias/vAdam/conv2d_48/kernel/vAdam/conv2d_48/bias/v#Adam/batch_normalization_60/gamma/v"Adam/batch_normalization_60/beta/v#Adam/batch_normalization_64/gamma/v"Adam/batch_normalization_64/beta/vAdam/conv2d_46/kernel/vAdam/conv2d_46/bias/vAdam/conv2d_49/kernel/vAdam/conv2d_49/bias/v#Adam/batch_normalization_61/gamma/v"Adam/batch_normalization_61/beta/v#Adam/batch_normalization_65/gamma/v"Adam/batch_normalization_65/beta/vAdam/conv2d_47/kernel/vAdam/conv2d_47/bias/vAdam/conv2d_50/kernel/vAdam/conv2d_50/bias/v#Adam/batch_normalization_62/gamma/v"Adam/batch_normalization_62/beta/v#Adam/batch_normalization_66/gamma/v"Adam/batch_normalization_66/beta/vAdam/dense_30/kernel/vAdam/dense_30/bias/vAdam/dense_32/kernel/vAdam/dense_32/bias/v#Adam/batch_normalization_63/gamma/v"Adam/batch_normalization_63/beta/v#Adam/batch_normalization_67/gamma/v"Adam/batch_normalization_67/beta/vAdam/dense_31/kernel/vAdam/dense_31/bias/vAdam/dense_33/kernel/vAdam/dense_33/bias/v*Ъ
TinТ
П2М*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_restore_43859э╓(
й
F
*__inference_flatten_15_layer_call_fn_42647

inputs
identity╟
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_15_layer_call_and_return_conditional_losses_396962
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
К
d
E__inference_dropout_63_layer_call_and_return_conditional_losses_39894

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Е
Ш
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42538

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
╖
F
*__inference_dropout_64_layer_call_fn_41900

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_64_layer_call_and_return_conditional_losses_390612
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         BB2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
Ї	
▌
D__inference_conv2d_45_layer_call_and_return_conditional_losses_38854

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpе
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞2	
BiasAddЯ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╞╞::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
∙
Ї
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_39219

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
╜
I
-__inference_activation_61_layer_call_fn_41948

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_61_layer_call_and_return_conditional_losses_391742
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB :W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
╝
й
6__inference_batch_normalization_67_layer_call_fn_42867

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_387702
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┴
Ї
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41692

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
й
F
*__inference_flatten_16_layer_call_fn_42658

inputs
identity╟
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_396822
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ї	
▌
D__inference_conv2d_48_layer_call_and_return_conditional_losses_38828

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpе
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞2	
BiasAddЯ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╞╞::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
Щ	
▄
C__inference_dense_32_layer_call_and_return_conditional_losses_42687

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
╜
a
E__inference_flatten_15_layer_call_and_return_conditional_losses_42642

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
┴
Ї
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_38267

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ъ	
▌
D__inference_conv2d_47_layer_call_and_return_conditional_losses_42278

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! 2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         !! ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
╓
d
H__inference_activation_66_layer_call_and_return_conditional_losses_39447

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         !! 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
┴
Ї
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41996

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_38364

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
р
й
6__inference_batch_normalization_60_layer_call_fn_41641

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_389882
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
╙
д
(__inference_face_net_layer_call_fn_41421

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity

identity_1ИвStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *F
_read_only_resource_inputs(
&$	
!"%&'(+,/01234*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_face_net_layer_call_and_return_conditional_losses_403002
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
ъ	
▌
D__inference_conv2d_49_layer_call_and_return_conditional_losses_41929

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB 2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         BB::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
в
й
6__inference_batch_normalization_66_layer_call_fn_42518

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_384992
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_38064

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╛
й
6__inference_batch_normalization_67_layer_call_fn_42880

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_388032
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
░
L
0__inference_max_pooling2d_45_layer_call_fn_38058

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_380522
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
├
d
E__inference_dropout_66_layer_call_and_return_conditional_losses_42621

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╘Ц
В/
C__inference_face_net_layer_call_and_return_conditional_losses_41102

inputs,
(conv2d_48_conv2d_readvariableop_resource-
)conv2d_48_biasadd_readvariableop_resource,
(conv2d_45_conv2d_readvariableop_resource-
)conv2d_45_biasadd_readvariableop_resource2
.batch_normalization_64_readvariableop_resource4
0batch_normalization_64_readvariableop_1_resourceC
?batch_normalization_64_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_64_fusedbatchnormv3_readvariableop_1_resource2
.batch_normalization_60_readvariableop_resource4
0batch_normalization_60_readvariableop_1_resourceC
?batch_normalization_60_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_60_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_49_conv2d_readvariableop_resource-
)conv2d_49_biasadd_readvariableop_resource,
(conv2d_46_conv2d_readvariableop_resource-
)conv2d_46_biasadd_readvariableop_resource2
.batch_normalization_65_readvariableop_resource4
0batch_normalization_65_readvariableop_1_resourceC
?batch_normalization_65_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_65_fusedbatchnormv3_readvariableop_1_resource2
.batch_normalization_61_readvariableop_resource4
0batch_normalization_61_readvariableop_1_resourceC
?batch_normalization_61_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_61_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_50_conv2d_readvariableop_resource-
)conv2d_50_biasadd_readvariableop_resource,
(conv2d_47_conv2d_readvariableop_resource-
)conv2d_47_biasadd_readvariableop_resource2
.batch_normalization_66_readvariableop_resource4
0batch_normalization_66_readvariableop_1_resourceC
?batch_normalization_66_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource2
.batch_normalization_62_readvariableop_resource4
0batch_normalization_62_readvariableop_1_resourceC
?batch_normalization_62_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_62_fusedbatchnormv3_readvariableop_1_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_30_matmul_readvariableop_resource,
(dense_30_biasadd_readvariableop_resource0
,batch_normalization_67_assignmovingavg_410152
.batch_normalization_67_assignmovingavg_1_41021@
<batch_normalization_67_batchnorm_mul_readvariableop_resource<
8batch_normalization_67_batchnorm_readvariableop_resource0
,batch_normalization_63_assignmovingavg_410472
.batch_normalization_63_assignmovingavg_1_41053@
<batch_normalization_63_batchnorm_mul_readvariableop_resource<
8batch_normalization_63_batchnorm_readvariableop_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource
identity

identity_1Ив%batch_normalization_60/AssignNewValueв'batch_normalization_60/AssignNewValue_1в6batch_normalization_60/FusedBatchNormV3/ReadVariableOpв8batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_60/ReadVariableOpв'batch_normalization_60/ReadVariableOp_1в%batch_normalization_61/AssignNewValueв'batch_normalization_61/AssignNewValue_1в6batch_normalization_61/FusedBatchNormV3/ReadVariableOpв8batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_61/ReadVariableOpв'batch_normalization_61/ReadVariableOp_1в%batch_normalization_62/AssignNewValueв'batch_normalization_62/AssignNewValue_1в6batch_normalization_62/FusedBatchNormV3/ReadVariableOpв8batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_62/ReadVariableOpв'batch_normalization_62/ReadVariableOp_1в:batch_normalization_63/AssignMovingAvg/AssignSubVariableOpв5batch_normalization_63/AssignMovingAvg/ReadVariableOpв<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOpв7batch_normalization_63/AssignMovingAvg_1/ReadVariableOpв/batch_normalization_63/batchnorm/ReadVariableOpв3batch_normalization_63/batchnorm/mul/ReadVariableOpв%batch_normalization_64/AssignNewValueв'batch_normalization_64/AssignNewValue_1в6batch_normalization_64/FusedBatchNormV3/ReadVariableOpв8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_64/ReadVariableOpв'batch_normalization_64/ReadVariableOp_1в%batch_normalization_65/AssignNewValueв'batch_normalization_65/AssignNewValue_1в6batch_normalization_65/FusedBatchNormV3/ReadVariableOpв8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_65/ReadVariableOpв'batch_normalization_65/ReadVariableOp_1в%batch_normalization_66/AssignNewValueв'batch_normalization_66/AssignNewValue_1в6batch_normalization_66/FusedBatchNormV3/ReadVariableOpв8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_66/ReadVariableOpв'batch_normalization_66/ReadVariableOp_1в:batch_normalization_67/AssignMovingAvg/AssignSubVariableOpв5batch_normalization_67/AssignMovingAvg/ReadVariableOpв<batch_normalization_67/AssignMovingAvg_1/AssignSubVariableOpв7batch_normalization_67/AssignMovingAvg_1/ReadVariableOpв/batch_normalization_67/batchnorm/ReadVariableOpв3batch_normalization_67/batchnorm/mul/ReadVariableOpв conv2d_45/BiasAdd/ReadVariableOpвconv2d_45/Conv2D/ReadVariableOpв conv2d_46/BiasAdd/ReadVariableOpвconv2d_46/Conv2D/ReadVariableOpв conv2d_47/BiasAdd/ReadVariableOpвconv2d_47/Conv2D/ReadVariableOpв conv2d_48/BiasAdd/ReadVariableOpвconv2d_48/Conv2D/ReadVariableOpв conv2d_49/BiasAdd/ReadVariableOpвconv2d_49/Conv2D/ReadVariableOpв conv2d_50/BiasAdd/ReadVariableOpвconv2d_50/Conv2D/ReadVariableOpвdense_30/BiasAdd/ReadVariableOpвdense_30/MatMul/ReadVariableOpвdense_31/BiasAdd/ReadVariableOpвdense_31/MatMul/ReadVariableOpвdense_32/BiasAdd/ReadVariableOpвdense_32/MatMul/ReadVariableOpвdense_33/BiasAdd/ReadVariableOpвdense_33/MatMul/ReadVariableOp│
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_48/Conv2D/ReadVariableOp├
conv2d_48/Conv2DConv2Dinputs'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞*
paddingSAME*
strides
2
conv2d_48/Conv2Dк
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp▓
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞2
conv2d_48/BiasAdd│
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_45/Conv2D/ReadVariableOp├
conv2d_45/Conv2DConv2Dinputs'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞*
paddingSAME*
strides
2
conv2d_45/Conv2Dк
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_45/BiasAdd/ReadVariableOp▓
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞2
conv2d_45/BiasAddИ
activation_64/ReluReluconv2d_48/BiasAdd:output:0*
T0*1
_output_shapes
:         ╞╞2
activation_64/ReluИ
activation_60/ReluReluconv2d_45/BiasAdd:output:0*
T0*1
_output_shapes
:         ╞╞2
activation_60/Relu╣
%batch_normalization_64/ReadVariableOpReadVariableOp.batch_normalization_64_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_64/ReadVariableOp┐
'batch_normalization_64/ReadVariableOp_1ReadVariableOp0batch_normalization_64_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_64/ReadVariableOp_1ь
6batch_normalization_64/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_64_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_64/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_64_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1■
'batch_normalization_64/FusedBatchNormV3FusedBatchNormV3 activation_64/Relu:activations:0-batch_normalization_64/ReadVariableOp:value:0/batch_normalization_64/ReadVariableOp_1:value:0>batch_normalization_64/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_64/FusedBatchNormV3╖
%batch_normalization_64/AssignNewValueAssignVariableOp?batch_normalization_64_fusedbatchnormv3_readvariableop_resource4batch_normalization_64/FusedBatchNormV3:batch_mean:07^batch_normalization_64/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_64/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_64/AssignNewValue┼
'batch_normalization_64/AssignNewValue_1AssignVariableOpAbatch_normalization_64_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_64/FusedBatchNormV3:batch_variance:09^batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_64/AssignNewValue_1╣
%batch_normalization_60/ReadVariableOpReadVariableOp.batch_normalization_60_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_60/ReadVariableOp┐
'batch_normalization_60/ReadVariableOp_1ReadVariableOp0batch_normalization_60_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_60/ReadVariableOp_1ь
6batch_normalization_60/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_60_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_60/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_60_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1■
'batch_normalization_60/FusedBatchNormV3FusedBatchNormV3 activation_60/Relu:activations:0-batch_normalization_60/ReadVariableOp:value:0/batch_normalization_60/ReadVariableOp_1:value:0>batch_normalization_60/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_60/FusedBatchNormV3╖
%batch_normalization_60/AssignNewValueAssignVariableOp?batch_normalization_60_fusedbatchnormv3_readvariableop_resource4batch_normalization_60/FusedBatchNormV3:batch_mean:07^batch_normalization_60/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_60/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_60/AssignNewValue┼
'batch_normalization_60/AssignNewValue_1AssignVariableOpAbatch_normalization_60_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_60/FusedBatchNormV3:batch_variance:09^batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_60/AssignNewValue_1┘
max_pooling2d_48/MaxPoolMaxPool+batch_normalization_64/FusedBatchNormV3:y:0*/
_output_shapes
:         BB*
ksize
*
paddingVALID*
strides
2
max_pooling2d_48/MaxPool┘
max_pooling2d_45/MaxPoolMaxPool+batch_normalization_60/FusedBatchNormV3:y:0*/
_output_shapes
:         BB*
ksize
*
paddingVALID*
strides
2
max_pooling2d_45/MaxPooly
dropout_64/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_64/dropout/Const╖
dropout_64/dropout/MulMul!max_pooling2d_48/MaxPool:output:0!dropout_64/dropout/Const:output:0*
T0*/
_output_shapes
:         BB2
dropout_64/dropout/MulЕ
dropout_64/dropout/ShapeShape!max_pooling2d_48/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_64/dropout/Shape▌
/dropout_64/dropout/random_uniform/RandomUniformRandomUniform!dropout_64/dropout/Shape:output:0*
T0*/
_output_shapes
:         BB*
dtype021
/dropout_64/dropout/random_uniform/RandomUniformЛ
!dropout_64/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_64/dropout/GreaterEqual/yЄ
dropout_64/dropout/GreaterEqualGreaterEqual8dropout_64/dropout/random_uniform/RandomUniform:output:0*dropout_64/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         BB2!
dropout_64/dropout/GreaterEqualи
dropout_64/dropout/CastCast#dropout_64/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         BB2
dropout_64/dropout/Castо
dropout_64/dropout/Mul_1Muldropout_64/dropout/Mul:z:0dropout_64/dropout/Cast:y:0*
T0*/
_output_shapes
:         BB2
dropout_64/dropout/Mul_1y
dropout_60/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_60/dropout/Const╖
dropout_60/dropout/MulMul!max_pooling2d_45/MaxPool:output:0!dropout_60/dropout/Const:output:0*
T0*/
_output_shapes
:         BB2
dropout_60/dropout/MulЕ
dropout_60/dropout/ShapeShape!max_pooling2d_45/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_60/dropout/Shape▌
/dropout_60/dropout/random_uniform/RandomUniformRandomUniform!dropout_60/dropout/Shape:output:0*
T0*/
_output_shapes
:         BB*
dtype021
/dropout_60/dropout/random_uniform/RandomUniformЛ
!dropout_60/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_60/dropout/GreaterEqual/yЄ
dropout_60/dropout/GreaterEqualGreaterEqual8dropout_60/dropout/random_uniform/RandomUniform:output:0*dropout_60/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         BB2!
dropout_60/dropout/GreaterEqualи
dropout_60/dropout/CastCast#dropout_60/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         BB2
dropout_60/dropout/Castо
dropout_60/dropout/Mul_1Muldropout_60/dropout/Mul:z:0dropout_60/dropout/Cast:y:0*
T0*/
_output_shapes
:         BB2
dropout_60/dropout/Mul_1│
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_49/Conv2D/ReadVariableOp╫
conv2d_49/Conv2DConv2Ddropout_64/dropout/Mul_1:z:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB *
paddingSAME*
strides
2
conv2d_49/Conv2Dк
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp░
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB 2
conv2d_49/BiasAdd│
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_46/Conv2D/ReadVariableOp╫
conv2d_46/Conv2DConv2Ddropout_60/dropout/Mul_1:z:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB *
paddingSAME*
strides
2
conv2d_46/Conv2Dк
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_46/BiasAdd/ReadVariableOp░
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB 2
conv2d_46/BiasAddЖ
activation_65/ReluReluconv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:         BB 2
activation_65/ReluЖ
activation_61/ReluReluconv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:         BB 2
activation_61/Relu╣
%batch_normalization_65/ReadVariableOpReadVariableOp.batch_normalization_65_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_65/ReadVariableOp┐
'batch_normalization_65/ReadVariableOp_1ReadVariableOp0batch_normalization_65_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_65/ReadVariableOp_1ь
6batch_normalization_65/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_65_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_65/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_65_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1№
'batch_normalization_65/FusedBatchNormV3FusedBatchNormV3 activation_65/Relu:activations:0-batch_normalization_65/ReadVariableOp:value:0/batch_normalization_65/ReadVariableOp_1:value:0>batch_normalization_65/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_65/FusedBatchNormV3╖
%batch_normalization_65/AssignNewValueAssignVariableOp?batch_normalization_65_fusedbatchnormv3_readvariableop_resource4batch_normalization_65/FusedBatchNormV3:batch_mean:07^batch_normalization_65/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_65/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_65/AssignNewValue┼
'batch_normalization_65/AssignNewValue_1AssignVariableOpAbatch_normalization_65_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_65/FusedBatchNormV3:batch_variance:09^batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_65/AssignNewValue_1╣
%batch_normalization_61/ReadVariableOpReadVariableOp.batch_normalization_61_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_61/ReadVariableOp┐
'batch_normalization_61/ReadVariableOp_1ReadVariableOp0batch_normalization_61_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_61/ReadVariableOp_1ь
6batch_normalization_61/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_61_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_61/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_61_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1№
'batch_normalization_61/FusedBatchNormV3FusedBatchNormV3 activation_61/Relu:activations:0-batch_normalization_61/ReadVariableOp:value:0/batch_normalization_61/ReadVariableOp_1:value:0>batch_normalization_61/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_61/FusedBatchNormV3╖
%batch_normalization_61/AssignNewValueAssignVariableOp?batch_normalization_61_fusedbatchnormv3_readvariableop_resource4batch_normalization_61/FusedBatchNormV3:batch_mean:07^batch_normalization_61/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_61/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_61/AssignNewValue┼
'batch_normalization_61/AssignNewValue_1AssignVariableOpAbatch_normalization_61_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_61/FusedBatchNormV3:batch_variance:09^batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_61/AssignNewValue_1┘
max_pooling2d_49/MaxPoolMaxPool+batch_normalization_65/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2
max_pooling2d_49/MaxPool┘
max_pooling2d_46/MaxPoolMaxPool+batch_normalization_61/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2
max_pooling2d_46/MaxPooly
dropout_65/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_65/dropout/Const╖
dropout_65/dropout/MulMul!max_pooling2d_49/MaxPool:output:0!dropout_65/dropout/Const:output:0*
T0*/
_output_shapes
:         !! 2
dropout_65/dropout/MulЕ
dropout_65/dropout/ShapeShape!max_pooling2d_49/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_65/dropout/Shape▌
/dropout_65/dropout/random_uniform/RandomUniformRandomUniform!dropout_65/dropout/Shape:output:0*
T0*/
_output_shapes
:         !! *
dtype021
/dropout_65/dropout/random_uniform/RandomUniformЛ
!dropout_65/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_65/dropout/GreaterEqual/yЄ
dropout_65/dropout/GreaterEqualGreaterEqual8dropout_65/dropout/random_uniform/RandomUniform:output:0*dropout_65/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         !! 2!
dropout_65/dropout/GreaterEqualи
dropout_65/dropout/CastCast#dropout_65/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         !! 2
dropout_65/dropout/Castо
dropout_65/dropout/Mul_1Muldropout_65/dropout/Mul:z:0dropout_65/dropout/Cast:y:0*
T0*/
_output_shapes
:         !! 2
dropout_65/dropout/Mul_1y
dropout_61/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_61/dropout/Const╖
dropout_61/dropout/MulMul!max_pooling2d_46/MaxPool:output:0!dropout_61/dropout/Const:output:0*
T0*/
_output_shapes
:         !! 2
dropout_61/dropout/MulЕ
dropout_61/dropout/ShapeShape!max_pooling2d_46/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_61/dropout/Shape▌
/dropout_61/dropout/random_uniform/RandomUniformRandomUniform!dropout_61/dropout/Shape:output:0*
T0*/
_output_shapes
:         !! *
dtype021
/dropout_61/dropout/random_uniform/RandomUniformЛ
!dropout_61/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_61/dropout/GreaterEqual/yЄ
dropout_61/dropout/GreaterEqualGreaterEqual8dropout_61/dropout/random_uniform/RandomUniform:output:0*dropout_61/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         !! 2!
dropout_61/dropout/GreaterEqualи
dropout_61/dropout/CastCast#dropout_61/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         !! 2
dropout_61/dropout/Castо
dropout_61/dropout/Mul_1Muldropout_61/dropout/Mul:z:0dropout_61/dropout/Cast:y:0*
T0*/
_output_shapes
:         !! 2
dropout_61/dropout/Mul_1│
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_50/Conv2D/ReadVariableOp╫
conv2d_50/Conv2DConv2Ddropout_65/dropout/Mul_1:z:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! *
paddingSAME*
strides
2
conv2d_50/Conv2Dк
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp░
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! 2
conv2d_50/BiasAdd│
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_47/Conv2D/ReadVariableOp╫
conv2d_47/Conv2DConv2Ddropout_61/dropout/Mul_1:z:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! *
paddingSAME*
strides
2
conv2d_47/Conv2Dк
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_47/BiasAdd/ReadVariableOp░
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! 2
conv2d_47/BiasAddЖ
activation_66/ReluReluconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:         !! 2
activation_66/ReluЖ
activation_62/ReluReluconv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:         !! 2
activation_62/Relu╣
%batch_normalization_66/ReadVariableOpReadVariableOp.batch_normalization_66_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_66/ReadVariableOp┐
'batch_normalization_66/ReadVariableOp_1ReadVariableOp0batch_normalization_66_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_66/ReadVariableOp_1ь
6batch_normalization_66/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_66_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_66/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1№
'batch_normalization_66/FusedBatchNormV3FusedBatchNormV3 activation_66/Relu:activations:0-batch_normalization_66/ReadVariableOp:value:0/batch_normalization_66/ReadVariableOp_1:value:0>batch_normalization_66/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_66/FusedBatchNormV3╖
%batch_normalization_66/AssignNewValueAssignVariableOp?batch_normalization_66_fusedbatchnormv3_readvariableop_resource4batch_normalization_66/FusedBatchNormV3:batch_mean:07^batch_normalization_66/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_66/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_66/AssignNewValue┼
'batch_normalization_66/AssignNewValue_1AssignVariableOpAbatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_66/FusedBatchNormV3:batch_variance:09^batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_66/AssignNewValue_1╣
%batch_normalization_62/ReadVariableOpReadVariableOp.batch_normalization_62_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_62/ReadVariableOp┐
'batch_normalization_62/ReadVariableOp_1ReadVariableOp0batch_normalization_62_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_62/ReadVariableOp_1ь
6batch_normalization_62/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_62_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_62/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_62_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1№
'batch_normalization_62/FusedBatchNormV3FusedBatchNormV3 activation_62/Relu:activations:0-batch_normalization_62/ReadVariableOp:value:0/batch_normalization_62/ReadVariableOp_1:value:0>batch_normalization_62/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2)
'batch_normalization_62/FusedBatchNormV3╖
%batch_normalization_62/AssignNewValueAssignVariableOp?batch_normalization_62_fusedbatchnormv3_readvariableop_resource4batch_normalization_62/FusedBatchNormV3:batch_mean:07^batch_normalization_62/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*R
_classH
FDloc:@batch_normalization_62/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02'
%batch_normalization_62/AssignNewValue┼
'batch_normalization_62/AssignNewValue_1AssignVariableOpAbatch_normalization_62_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_62/FusedBatchNormV3:batch_variance:09^batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*T
_classJ
HFloc:@batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02)
'batch_normalization_62/AssignNewValue_1┘
max_pooling2d_50/MaxPoolMaxPool+batch_normalization_66/FusedBatchNormV3:y:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_50/MaxPool┘
max_pooling2d_47/MaxPoolMaxPool+batch_normalization_62/FusedBatchNormV3:y:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_47/MaxPooly
dropout_66/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_66/dropout/Const╖
dropout_66/dropout/MulMul!max_pooling2d_50/MaxPool:output:0!dropout_66/dropout/Const:output:0*
T0*/
_output_shapes
:          2
dropout_66/dropout/MulЕ
dropout_66/dropout/ShapeShape!max_pooling2d_50/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_66/dropout/Shape▌
/dropout_66/dropout/random_uniform/RandomUniformRandomUniform!dropout_66/dropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype021
/dropout_66/dropout/random_uniform/RandomUniformЛ
!dropout_66/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_66/dropout/GreaterEqual/yЄ
dropout_66/dropout/GreaterEqualGreaterEqual8dropout_66/dropout/random_uniform/RandomUniform:output:0*dropout_66/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          2!
dropout_66/dropout/GreaterEqualи
dropout_66/dropout/CastCast#dropout_66/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          2
dropout_66/dropout/Castо
dropout_66/dropout/Mul_1Muldropout_66/dropout/Mul:z:0dropout_66/dropout/Cast:y:0*
T0*/
_output_shapes
:          2
dropout_66/dropout/Mul_1y
dropout_62/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_62/dropout/Const╖
dropout_62/dropout/MulMul!max_pooling2d_47/MaxPool:output:0!dropout_62/dropout/Const:output:0*
T0*/
_output_shapes
:          2
dropout_62/dropout/MulЕ
dropout_62/dropout/ShapeShape!max_pooling2d_47/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_62/dropout/Shape▌
/dropout_62/dropout/random_uniform/RandomUniformRandomUniform!dropout_62/dropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype021
/dropout_62/dropout/random_uniform/RandomUniformЛ
!dropout_62/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_62/dropout/GreaterEqual/yЄ
dropout_62/dropout/GreaterEqualGreaterEqual8dropout_62/dropout/random_uniform/RandomUniform:output:0*dropout_62/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          2!
dropout_62/dropout/GreaterEqualи
dropout_62/dropout/CastCast#dropout_62/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          2
dropout_62/dropout/Castо
dropout_62/dropout/Mul_1Muldropout_62/dropout/Mul:z:0dropout_62/dropout/Cast:y:0*
T0*/
_output_shapes
:          2
dropout_62/dropout/Mul_1u
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_16/ConstЯ
flatten_16/ReshapeReshapedropout_66/dropout/Mul_1:z:0flatten_16/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_16/Reshapeu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_15/ConstЯ
flatten_15/ReshapeReshapedropout_62/dropout/Mul_1:z:0flatten_15/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_15/Reshapeк
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_32/MatMul/ReadVariableOpд
dense_32/MatMulMatMulflatten_16/Reshape:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_32/MatMulи
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_32/BiasAdd/ReadVariableOpж
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_32/BiasAddк
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_30/MatMul/ReadVariableOpд
dense_30/MatMulMatMulflatten_15/Reshape:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_30/MatMulи
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_30/BiasAdd/ReadVariableOpж
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_30/BiasAdd~
activation_67/ReluReludense_32/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
activation_67/Relu~
activation_63/ReluReludense_30/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
activation_63/Relu╕
5batch_normalization_67/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_67/moments/mean/reduction_indicesя
#batch_normalization_67/moments/meanMean activation_67/Relu:activations:0>batch_normalization_67/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2%
#batch_normalization_67/moments/mean┬
+batch_normalization_67/moments/StopGradientStopGradient,batch_normalization_67/moments/mean:output:0*
T0*
_output_shapes
:	А2-
+batch_normalization_67/moments/StopGradientД
0batch_normalization_67/moments/SquaredDifferenceSquaredDifference activation_67/Relu:activations:04batch_normalization_67/moments/StopGradient:output:0*
T0*(
_output_shapes
:         А22
0batch_normalization_67/moments/SquaredDifference└
9batch_normalization_67/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_67/moments/variance/reduction_indicesП
'batch_normalization_67/moments/varianceMean4batch_normalization_67/moments/SquaredDifference:z:0Bbatch_normalization_67/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2)
'batch_normalization_67/moments/variance╞
&batch_normalization_67/moments/SqueezeSqueeze,batch_normalization_67/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2(
&batch_normalization_67/moments/Squeeze╬
(batch_normalization_67/moments/Squeeze_1Squeeze0batch_normalization_67/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2*
(batch_normalization_67/moments/Squeeze_1Р
,batch_normalization_67/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_67/AssignMovingAvg/41015*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2.
,batch_normalization_67/AssignMovingAvg/decay╪
5batch_normalization_67/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_67_assignmovingavg_41015*
_output_shapes	
:А*
dtype027
5batch_normalization_67/AssignMovingAvg/ReadVariableOpф
*batch_normalization_67/AssignMovingAvg/subSub=batch_normalization_67/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_67/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_67/AssignMovingAvg/41015*
_output_shapes	
:А2,
*batch_normalization_67/AssignMovingAvg/sub█
*batch_normalization_67/AssignMovingAvg/mulMul.batch_normalization_67/AssignMovingAvg/sub:z:05batch_normalization_67/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_67/AssignMovingAvg/41015*
_output_shapes	
:А2,
*batch_normalization_67/AssignMovingAvg/mul╖
:batch_normalization_67/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_67_assignmovingavg_41015.batch_normalization_67/AssignMovingAvg/mul:z:06^batch_normalization_67/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_67/AssignMovingAvg/41015*
_output_shapes
 *
dtype02<
:batch_normalization_67/AssignMovingAvg/AssignSubVariableOpЦ
.batch_normalization_67/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_67/AssignMovingAvg_1/41021*
_output_shapes
: *
dtype0*
valueB
 *
╫#<20
.batch_normalization_67/AssignMovingAvg_1/decay▐
7batch_normalization_67/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_67_assignmovingavg_1_41021*
_output_shapes	
:А*
dtype029
7batch_normalization_67/AssignMovingAvg_1/ReadVariableOpю
,batch_normalization_67/AssignMovingAvg_1/subSub?batch_normalization_67/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_67/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_67/AssignMovingAvg_1/41021*
_output_shapes	
:А2.
,batch_normalization_67/AssignMovingAvg_1/subх
,batch_normalization_67/AssignMovingAvg_1/mulMul0batch_normalization_67/AssignMovingAvg_1/sub:z:07batch_normalization_67/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_67/AssignMovingAvg_1/41021*
_output_shapes	
:А2.
,batch_normalization_67/AssignMovingAvg_1/mul├
<batch_normalization_67/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_67_assignmovingavg_1_410210batch_normalization_67/AssignMovingAvg_1/mul:z:08^batch_normalization_67/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_67/AssignMovingAvg_1/41021*
_output_shapes
 *
dtype02>
<batch_normalization_67/AssignMovingAvg_1/AssignSubVariableOpХ
&batch_normalization_67/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2(
&batch_normalization_67/batchnorm/add/y▀
$batch_normalization_67/batchnorm/addAddV21batch_normalization_67/moments/Squeeze_1:output:0/batch_normalization_67/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2&
$batch_normalization_67/batchnorm/addй
&batch_normalization_67/batchnorm/RsqrtRsqrt(batch_normalization_67/batchnorm/add:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_67/batchnorm/Rsqrtф
3batch_normalization_67/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_67_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3batch_normalization_67/batchnorm/mul/ReadVariableOpт
$batch_normalization_67/batchnorm/mulMul*batch_normalization_67/batchnorm/Rsqrt:y:0;batch_normalization_67/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2&
$batch_normalization_67/batchnorm/mul╓
&batch_normalization_67/batchnorm/mul_1Mul activation_67/Relu:activations:0(batch_normalization_67/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_67/batchnorm/mul_1╪
&batch_normalization_67/batchnorm/mul_2Mul/batch_normalization_67/moments/Squeeze:output:0(batch_normalization_67/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_67/batchnorm/mul_2╪
/batch_normalization_67/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_67_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/batch_normalization_67/batchnorm/ReadVariableOp▐
$batch_normalization_67/batchnorm/subSub7batch_normalization_67/batchnorm/ReadVariableOp:value:0*batch_normalization_67/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2&
$batch_normalization_67/batchnorm/subт
&batch_normalization_67/batchnorm/add_1AddV2*batch_normalization_67/batchnorm/mul_1:z:0(batch_normalization_67/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_67/batchnorm/add_1╕
5batch_normalization_63/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 27
5batch_normalization_63/moments/mean/reduction_indicesя
#batch_normalization_63/moments/meanMean activation_63/Relu:activations:0>batch_normalization_63/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2%
#batch_normalization_63/moments/mean┬
+batch_normalization_63/moments/StopGradientStopGradient,batch_normalization_63/moments/mean:output:0*
T0*
_output_shapes
:	А2-
+batch_normalization_63/moments/StopGradientД
0batch_normalization_63/moments/SquaredDifferenceSquaredDifference activation_63/Relu:activations:04batch_normalization_63/moments/StopGradient:output:0*
T0*(
_output_shapes
:         А22
0batch_normalization_63/moments/SquaredDifference└
9batch_normalization_63/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2;
9batch_normalization_63/moments/variance/reduction_indicesП
'batch_normalization_63/moments/varianceMean4batch_normalization_63/moments/SquaredDifference:z:0Bbatch_normalization_63/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2)
'batch_normalization_63/moments/variance╞
&batch_normalization_63/moments/SqueezeSqueeze,batch_normalization_63/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2(
&batch_normalization_63/moments/Squeeze╬
(batch_normalization_63/moments/Squeeze_1Squeeze0batch_normalization_63/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2*
(batch_normalization_63/moments/Squeeze_1Р
,batch_normalization_63/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_63/AssignMovingAvg/41047*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2.
,batch_normalization_63/AssignMovingAvg/decay╪
5batch_normalization_63/AssignMovingAvg/ReadVariableOpReadVariableOp,batch_normalization_63_assignmovingavg_41047*
_output_shapes	
:А*
dtype027
5batch_normalization_63/AssignMovingAvg/ReadVariableOpф
*batch_normalization_63/AssignMovingAvg/subSub=batch_normalization_63/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_63/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_63/AssignMovingAvg/41047*
_output_shapes	
:А2,
*batch_normalization_63/AssignMovingAvg/sub█
*batch_normalization_63/AssignMovingAvg/mulMul.batch_normalization_63/AssignMovingAvg/sub:z:05batch_normalization_63/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*?
_class5
31loc:@batch_normalization_63/AssignMovingAvg/41047*
_output_shapes	
:А2,
*batch_normalization_63/AssignMovingAvg/mul╖
:batch_normalization_63/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,batch_normalization_63_assignmovingavg_41047.batch_normalization_63/AssignMovingAvg/mul:z:06^batch_normalization_63/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*?
_class5
31loc:@batch_normalization_63/AssignMovingAvg/41047*
_output_shapes
 *
dtype02<
:batch_normalization_63/AssignMovingAvg/AssignSubVariableOpЦ
.batch_normalization_63/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_63/AssignMovingAvg_1/41053*
_output_shapes
: *
dtype0*
valueB
 *
╫#<20
.batch_normalization_63/AssignMovingAvg_1/decay▐
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOpReadVariableOp.batch_normalization_63_assignmovingavg_1_41053*
_output_shapes	
:А*
dtype029
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOpю
,batch_normalization_63/AssignMovingAvg_1/subSub?batch_normalization_63/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_63/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_63/AssignMovingAvg_1/41053*
_output_shapes	
:А2.
,batch_normalization_63/AssignMovingAvg_1/subх
,batch_normalization_63/AssignMovingAvg_1/mulMul0batch_normalization_63/AssignMovingAvg_1/sub:z:07batch_normalization_63/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*A
_class7
53loc:@batch_normalization_63/AssignMovingAvg_1/41053*
_output_shapes	
:А2.
,batch_normalization_63/AssignMovingAvg_1/mul├
<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.batch_normalization_63_assignmovingavg_1_410530batch_normalization_63/AssignMovingAvg_1/mul:z:08^batch_normalization_63/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*A
_class7
53loc:@batch_normalization_63/AssignMovingAvg_1/41053*
_output_shapes
 *
dtype02>
<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOpХ
&batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2(
&batch_normalization_63/batchnorm/add/y▀
$batch_normalization_63/batchnorm/addAddV21batch_normalization_63/moments/Squeeze_1:output:0/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2&
$batch_normalization_63/batchnorm/addй
&batch_normalization_63/batchnorm/RsqrtRsqrt(batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_63/batchnorm/Rsqrtф
3batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3batch_normalization_63/batchnorm/mul/ReadVariableOpт
$batch_normalization_63/batchnorm/mulMul*batch_normalization_63/batchnorm/Rsqrt:y:0;batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2&
$batch_normalization_63/batchnorm/mul╓
&batch_normalization_63/batchnorm/mul_1Mul activation_63/Relu:activations:0(batch_normalization_63/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_63/batchnorm/mul_1╪
&batch_normalization_63/batchnorm/mul_2Mul/batch_normalization_63/moments/Squeeze:output:0(batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_63/batchnorm/mul_2╪
/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/batch_normalization_63/batchnorm/ReadVariableOp▐
$batch_normalization_63/batchnorm/subSub7batch_normalization_63/batchnorm/ReadVariableOp:value:0*batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2&
$batch_normalization_63/batchnorm/subт
&batch_normalization_63/batchnorm/add_1AddV2*batch_normalization_63/batchnorm/mul_1:z:0(batch_normalization_63/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_63/batchnorm/add_1y
dropout_67/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_67/dropout/Const╣
dropout_67/dropout/MulMul*batch_normalization_67/batchnorm/add_1:z:0!dropout_67/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_67/dropout/MulО
dropout_67/dropout/ShapeShape*batch_normalization_67/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_67/dropout/Shape╓
/dropout_67/dropout/random_uniform/RandomUniformRandomUniform!dropout_67/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_67/dropout/random_uniform/RandomUniformЛ
!dropout_67/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_67/dropout/GreaterEqual/yы
dropout_67/dropout/GreaterEqualGreaterEqual8dropout_67/dropout/random_uniform/RandomUniform:output:0*dropout_67/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_67/dropout/GreaterEqualб
dropout_67/dropout/CastCast#dropout_67/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_67/dropout/Castз
dropout_67/dropout/Mul_1Muldropout_67/dropout/Mul:z:0dropout_67/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_67/dropout/Mul_1y
dropout_63/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_63/dropout/Const╣
dropout_63/dropout/MulMul*batch_normalization_63/batchnorm/add_1:z:0!dropout_63/dropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout_63/dropout/MulО
dropout_63/dropout/ShapeShape*batch_normalization_63/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_63/dropout/Shape╓
/dropout_63/dropout/random_uniform/RandomUniformRandomUniform!dropout_63/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype021
/dropout_63/dropout/random_uniform/RandomUniformЛ
!dropout_63/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_63/dropout/GreaterEqual/yы
dropout_63/dropout/GreaterEqualGreaterEqual8dropout_63/dropout/random_uniform/RandomUniform:output:0*dropout_63/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2!
dropout_63/dropout/GreaterEqualб
dropout_63/dropout/CastCast#dropout_63/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout_63/dropout/Castз
dropout_63/dropout/Mul_1Muldropout_63/dropout/Mul:z:0dropout_63/dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout_63/dropout/Mul_1й
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_33/MatMul/ReadVariableOpд
dense_33/MatMulMatMuldropout_67/dropout/Mul_1:z:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_33/MatMulз
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_33/BiasAdd/ReadVariableOpе
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_33/BiasAddй
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_31/MatMul/ReadVariableOpд
dense_31/MatMulMatMuldropout_63/dropout/Mul_1:z:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_31/MatMulз
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_31/BiasAdd/ReadVariableOpе
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_31/BiasAddЖ
gender_output/SigmoidSigmoiddense_33/BiasAdd:output:0*
T0*'
_output_shapes
:         2
gender_output/Sigmoid╙
IdentityIdentitydense_31/BiasAdd:output:0&^batch_normalization_60/AssignNewValue(^batch_normalization_60/AssignNewValue_17^batch_normalization_60/FusedBatchNormV3/ReadVariableOp9^batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_60/ReadVariableOp(^batch_normalization_60/ReadVariableOp_1&^batch_normalization_61/AssignNewValue(^batch_normalization_61/AssignNewValue_17^batch_normalization_61/FusedBatchNormV3/ReadVariableOp9^batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_61/ReadVariableOp(^batch_normalization_61/ReadVariableOp_1&^batch_normalization_62/AssignNewValue(^batch_normalization_62/AssignNewValue_17^batch_normalization_62/FusedBatchNormV3/ReadVariableOp9^batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_62/ReadVariableOp(^batch_normalization_62/ReadVariableOp_1;^batch_normalization_63/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_63/AssignMovingAvg/ReadVariableOp=^batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_63/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_63/batchnorm/ReadVariableOp4^batch_normalization_63/batchnorm/mul/ReadVariableOp&^batch_normalization_64/AssignNewValue(^batch_normalization_64/AssignNewValue_17^batch_normalization_64/FusedBatchNormV3/ReadVariableOp9^batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_64/ReadVariableOp(^batch_normalization_64/ReadVariableOp_1&^batch_normalization_65/AssignNewValue(^batch_normalization_65/AssignNewValue_17^batch_normalization_65/FusedBatchNormV3/ReadVariableOp9^batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_65/ReadVariableOp(^batch_normalization_65/ReadVariableOp_1&^batch_normalization_66/AssignNewValue(^batch_normalization_66/AssignNewValue_17^batch_normalization_66/FusedBatchNormV3/ReadVariableOp9^batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_66/ReadVariableOp(^batch_normalization_66/ReadVariableOp_1;^batch_normalization_67/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_67/AssignMovingAvg/ReadVariableOp=^batch_normalization_67/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_67/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_67/batchnorm/ReadVariableOp4^batch_normalization_67/batchnorm/mul/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity╫

Identity_1Identitygender_output/Sigmoid:y:0&^batch_normalization_60/AssignNewValue(^batch_normalization_60/AssignNewValue_17^batch_normalization_60/FusedBatchNormV3/ReadVariableOp9^batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_60/ReadVariableOp(^batch_normalization_60/ReadVariableOp_1&^batch_normalization_61/AssignNewValue(^batch_normalization_61/AssignNewValue_17^batch_normalization_61/FusedBatchNormV3/ReadVariableOp9^batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_61/ReadVariableOp(^batch_normalization_61/ReadVariableOp_1&^batch_normalization_62/AssignNewValue(^batch_normalization_62/AssignNewValue_17^batch_normalization_62/FusedBatchNormV3/ReadVariableOp9^batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_62/ReadVariableOp(^batch_normalization_62/ReadVariableOp_1;^batch_normalization_63/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_63/AssignMovingAvg/ReadVariableOp=^batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_63/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_63/batchnorm/ReadVariableOp4^batch_normalization_63/batchnorm/mul/ReadVariableOp&^batch_normalization_64/AssignNewValue(^batch_normalization_64/AssignNewValue_17^batch_normalization_64/FusedBatchNormV3/ReadVariableOp9^batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_64/ReadVariableOp(^batch_normalization_64/ReadVariableOp_1&^batch_normalization_65/AssignNewValue(^batch_normalization_65/AssignNewValue_17^batch_normalization_65/FusedBatchNormV3/ReadVariableOp9^batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_65/ReadVariableOp(^batch_normalization_65/ReadVariableOp_1&^batch_normalization_66/AssignNewValue(^batch_normalization_66/AssignNewValue_17^batch_normalization_66/FusedBatchNormV3/ReadVariableOp9^batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_66/ReadVariableOp(^batch_normalization_66/ReadVariableOp_1;^batch_normalization_67/AssignMovingAvg/AssignSubVariableOp6^batch_normalization_67/AssignMovingAvg/ReadVariableOp=^batch_normalization_67/AssignMovingAvg_1/AssignSubVariableOp8^batch_normalization_67/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_67/batchnorm/ReadVariableOp4^batch_normalization_67/batchnorm/mul/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::2N
%batch_normalization_60/AssignNewValue%batch_normalization_60/AssignNewValue2R
'batch_normalization_60/AssignNewValue_1'batch_normalization_60/AssignNewValue_12p
6batch_normalization_60/FusedBatchNormV3/ReadVariableOp6batch_normalization_60/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_60/FusedBatchNormV3/ReadVariableOp_18batch_normalization_60/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_60/ReadVariableOp%batch_normalization_60/ReadVariableOp2R
'batch_normalization_60/ReadVariableOp_1'batch_normalization_60/ReadVariableOp_12N
%batch_normalization_61/AssignNewValue%batch_normalization_61/AssignNewValue2R
'batch_normalization_61/AssignNewValue_1'batch_normalization_61/AssignNewValue_12p
6batch_normalization_61/FusedBatchNormV3/ReadVariableOp6batch_normalization_61/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_61/FusedBatchNormV3/ReadVariableOp_18batch_normalization_61/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_61/ReadVariableOp%batch_normalization_61/ReadVariableOp2R
'batch_normalization_61/ReadVariableOp_1'batch_normalization_61/ReadVariableOp_12N
%batch_normalization_62/AssignNewValue%batch_normalization_62/AssignNewValue2R
'batch_normalization_62/AssignNewValue_1'batch_normalization_62/AssignNewValue_12p
6batch_normalization_62/FusedBatchNormV3/ReadVariableOp6batch_normalization_62/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_62/FusedBatchNormV3/ReadVariableOp_18batch_normalization_62/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_62/ReadVariableOp%batch_normalization_62/ReadVariableOp2R
'batch_normalization_62/ReadVariableOp_1'batch_normalization_62/ReadVariableOp_12x
:batch_normalization_63/AssignMovingAvg/AssignSubVariableOp:batch_normalization_63/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_63/AssignMovingAvg/ReadVariableOp5batch_normalization_63/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_63/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp7batch_normalization_63/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_63/batchnorm/ReadVariableOp/batch_normalization_63/batchnorm/ReadVariableOp2j
3batch_normalization_63/batchnorm/mul/ReadVariableOp3batch_normalization_63/batchnorm/mul/ReadVariableOp2N
%batch_normalization_64/AssignNewValue%batch_normalization_64/AssignNewValue2R
'batch_normalization_64/AssignNewValue_1'batch_normalization_64/AssignNewValue_12p
6batch_normalization_64/FusedBatchNormV3/ReadVariableOp6batch_normalization_64/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_18batch_normalization_64/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_64/ReadVariableOp%batch_normalization_64/ReadVariableOp2R
'batch_normalization_64/ReadVariableOp_1'batch_normalization_64/ReadVariableOp_12N
%batch_normalization_65/AssignNewValue%batch_normalization_65/AssignNewValue2R
'batch_normalization_65/AssignNewValue_1'batch_normalization_65/AssignNewValue_12p
6batch_normalization_65/FusedBatchNormV3/ReadVariableOp6batch_normalization_65/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_18batch_normalization_65/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_65/ReadVariableOp%batch_normalization_65/ReadVariableOp2R
'batch_normalization_65/ReadVariableOp_1'batch_normalization_65/ReadVariableOp_12N
%batch_normalization_66/AssignNewValue%batch_normalization_66/AssignNewValue2R
'batch_normalization_66/AssignNewValue_1'batch_normalization_66/AssignNewValue_12p
6batch_normalization_66/FusedBatchNormV3/ReadVariableOp6batch_normalization_66/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_18batch_normalization_66/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_66/ReadVariableOp%batch_normalization_66/ReadVariableOp2R
'batch_normalization_66/ReadVariableOp_1'batch_normalization_66/ReadVariableOp_12x
:batch_normalization_67/AssignMovingAvg/AssignSubVariableOp:batch_normalization_67/AssignMovingAvg/AssignSubVariableOp2n
5batch_normalization_67/AssignMovingAvg/ReadVariableOp5batch_normalization_67/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_67/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_67/AssignMovingAvg_1/AssignSubVariableOp2r
7batch_normalization_67/AssignMovingAvg_1/ReadVariableOp7batch_normalization_67/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_67/batchnorm/ReadVariableOp/batch_normalization_67/batchnorm/ReadVariableOp2j
3batch_normalization_67/batchnorm/mul/ReadVariableOp3batch_normalization_67/batchnorm/mul/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
ды
─0
 __inference__wrapped_model_37838
input_65
1face_net_conv2d_48_conv2d_readvariableop_resource6
2face_net_conv2d_48_biasadd_readvariableop_resource5
1face_net_conv2d_45_conv2d_readvariableop_resource6
2face_net_conv2d_45_biasadd_readvariableop_resource;
7face_net_batch_normalization_64_readvariableop_resource=
9face_net_batch_normalization_64_readvariableop_1_resourceL
Hface_net_batch_normalization_64_fusedbatchnormv3_readvariableop_resourceN
Jface_net_batch_normalization_64_fusedbatchnormv3_readvariableop_1_resource;
7face_net_batch_normalization_60_readvariableop_resource=
9face_net_batch_normalization_60_readvariableop_1_resourceL
Hface_net_batch_normalization_60_fusedbatchnormv3_readvariableop_resourceN
Jface_net_batch_normalization_60_fusedbatchnormv3_readvariableop_1_resource5
1face_net_conv2d_49_conv2d_readvariableop_resource6
2face_net_conv2d_49_biasadd_readvariableop_resource5
1face_net_conv2d_46_conv2d_readvariableop_resource6
2face_net_conv2d_46_biasadd_readvariableop_resource;
7face_net_batch_normalization_65_readvariableop_resource=
9face_net_batch_normalization_65_readvariableop_1_resourceL
Hface_net_batch_normalization_65_fusedbatchnormv3_readvariableop_resourceN
Jface_net_batch_normalization_65_fusedbatchnormv3_readvariableop_1_resource;
7face_net_batch_normalization_61_readvariableop_resource=
9face_net_batch_normalization_61_readvariableop_1_resourceL
Hface_net_batch_normalization_61_fusedbatchnormv3_readvariableop_resourceN
Jface_net_batch_normalization_61_fusedbatchnormv3_readvariableop_1_resource5
1face_net_conv2d_50_conv2d_readvariableop_resource6
2face_net_conv2d_50_biasadd_readvariableop_resource5
1face_net_conv2d_47_conv2d_readvariableop_resource6
2face_net_conv2d_47_biasadd_readvariableop_resource;
7face_net_batch_normalization_66_readvariableop_resource=
9face_net_batch_normalization_66_readvariableop_1_resourceL
Hface_net_batch_normalization_66_fusedbatchnormv3_readvariableop_resourceN
Jface_net_batch_normalization_66_fusedbatchnormv3_readvariableop_1_resource;
7face_net_batch_normalization_62_readvariableop_resource=
9face_net_batch_normalization_62_readvariableop_1_resourceL
Hface_net_batch_normalization_62_fusedbatchnormv3_readvariableop_resourceN
Jface_net_batch_normalization_62_fusedbatchnormv3_readvariableop_1_resource4
0face_net_dense_32_matmul_readvariableop_resource5
1face_net_dense_32_biasadd_readvariableop_resource4
0face_net_dense_30_matmul_readvariableop_resource5
1face_net_dense_30_biasadd_readvariableop_resourceE
Aface_net_batch_normalization_67_batchnorm_readvariableop_resourceI
Eface_net_batch_normalization_67_batchnorm_mul_readvariableop_resourceG
Cface_net_batch_normalization_67_batchnorm_readvariableop_1_resourceG
Cface_net_batch_normalization_67_batchnorm_readvariableop_2_resourceE
Aface_net_batch_normalization_63_batchnorm_readvariableop_resourceI
Eface_net_batch_normalization_63_batchnorm_mul_readvariableop_resourceG
Cface_net_batch_normalization_63_batchnorm_readvariableop_1_resourceG
Cface_net_batch_normalization_63_batchnorm_readvariableop_2_resource4
0face_net_dense_33_matmul_readvariableop_resource5
1face_net_dense_33_biasadd_readvariableop_resource4
0face_net_dense_31_matmul_readvariableop_resource5
1face_net_dense_31_biasadd_readvariableop_resource
identity

identity_1Ив?face_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOpвAface_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1в.face_net/batch_normalization_60/ReadVariableOpв0face_net/batch_normalization_60/ReadVariableOp_1в?face_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOpвAface_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1в.face_net/batch_normalization_61/ReadVariableOpв0face_net/batch_normalization_61/ReadVariableOp_1в?face_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOpвAface_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1в.face_net/batch_normalization_62/ReadVariableOpв0face_net/batch_normalization_62/ReadVariableOp_1в8face_net/batch_normalization_63/batchnorm/ReadVariableOpв:face_net/batch_normalization_63/batchnorm/ReadVariableOp_1в:face_net/batch_normalization_63/batchnorm/ReadVariableOp_2в<face_net/batch_normalization_63/batchnorm/mul/ReadVariableOpв?face_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOpвAface_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1в.face_net/batch_normalization_64/ReadVariableOpв0face_net/batch_normalization_64/ReadVariableOp_1в?face_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOpвAface_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1в.face_net/batch_normalization_65/ReadVariableOpв0face_net/batch_normalization_65/ReadVariableOp_1в?face_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOpвAface_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1в.face_net/batch_normalization_66/ReadVariableOpв0face_net/batch_normalization_66/ReadVariableOp_1в8face_net/batch_normalization_67/batchnorm/ReadVariableOpв:face_net/batch_normalization_67/batchnorm/ReadVariableOp_1в:face_net/batch_normalization_67/batchnorm/ReadVariableOp_2в<face_net/batch_normalization_67/batchnorm/mul/ReadVariableOpв)face_net/conv2d_45/BiasAdd/ReadVariableOpв(face_net/conv2d_45/Conv2D/ReadVariableOpв)face_net/conv2d_46/BiasAdd/ReadVariableOpв(face_net/conv2d_46/Conv2D/ReadVariableOpв)face_net/conv2d_47/BiasAdd/ReadVariableOpв(face_net/conv2d_47/Conv2D/ReadVariableOpв)face_net/conv2d_48/BiasAdd/ReadVariableOpв(face_net/conv2d_48/Conv2D/ReadVariableOpв)face_net/conv2d_49/BiasAdd/ReadVariableOpв(face_net/conv2d_49/Conv2D/ReadVariableOpв)face_net/conv2d_50/BiasAdd/ReadVariableOpв(face_net/conv2d_50/Conv2D/ReadVariableOpв(face_net/dense_30/BiasAdd/ReadVariableOpв'face_net/dense_30/MatMul/ReadVariableOpв(face_net/dense_31/BiasAdd/ReadVariableOpв'face_net/dense_31/MatMul/ReadVariableOpв(face_net/dense_32/BiasAdd/ReadVariableOpв'face_net/dense_32/MatMul/ReadVariableOpв(face_net/dense_33/BiasAdd/ReadVariableOpв'face_net/dense_33/MatMul/ReadVariableOp╬
(face_net/conv2d_48/Conv2D/ReadVariableOpReadVariableOp1face_net_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(face_net/conv2d_48/Conv2D/ReadVariableOp▀
face_net/conv2d_48/Conv2DConv2Dinput_60face_net/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞*
paddingSAME*
strides
2
face_net/conv2d_48/Conv2D┼
)face_net/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp2face_net_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)face_net/conv2d_48/BiasAdd/ReadVariableOp╓
face_net/conv2d_48/BiasAddBiasAdd"face_net/conv2d_48/Conv2D:output:01face_net/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞2
face_net/conv2d_48/BiasAdd╬
(face_net/conv2d_45/Conv2D/ReadVariableOpReadVariableOp1face_net_conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(face_net/conv2d_45/Conv2D/ReadVariableOp▀
face_net/conv2d_45/Conv2DConv2Dinput_60face_net/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞*
paddingSAME*
strides
2
face_net/conv2d_45/Conv2D┼
)face_net/conv2d_45/BiasAdd/ReadVariableOpReadVariableOp2face_net_conv2d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)face_net/conv2d_45/BiasAdd/ReadVariableOp╓
face_net/conv2d_45/BiasAddBiasAdd"face_net/conv2d_45/Conv2D:output:01face_net/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞2
face_net/conv2d_45/BiasAddг
face_net/activation_64/ReluRelu#face_net/conv2d_48/BiasAdd:output:0*
T0*1
_output_shapes
:         ╞╞2
face_net/activation_64/Reluг
face_net/activation_60/ReluRelu#face_net/conv2d_45/BiasAdd:output:0*
T0*1
_output_shapes
:         ╞╞2
face_net/activation_60/Relu╘
.face_net/batch_normalization_64/ReadVariableOpReadVariableOp7face_net_batch_normalization_64_readvariableop_resource*
_output_shapes
:*
dtype020
.face_net/batch_normalization_64/ReadVariableOp┌
0face_net/batch_normalization_64/ReadVariableOp_1ReadVariableOp9face_net_batch_normalization_64_readvariableop_1_resource*
_output_shapes
:*
dtype022
0face_net/batch_normalization_64/ReadVariableOp_1З
?face_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOpReadVariableOpHface_net_batch_normalization_64_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?face_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOpН
Aface_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJface_net_batch_normalization_64_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Aface_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1п
0face_net/batch_normalization_64/FusedBatchNormV3FusedBatchNormV3)face_net/activation_64/Relu:activations:06face_net/batch_normalization_64/ReadVariableOp:value:08face_net/batch_normalization_64/ReadVariableOp_1:value:0Gface_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOp:value:0Iface_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
is_training( 22
0face_net/batch_normalization_64/FusedBatchNormV3╘
.face_net/batch_normalization_60/ReadVariableOpReadVariableOp7face_net_batch_normalization_60_readvariableop_resource*
_output_shapes
:*
dtype020
.face_net/batch_normalization_60/ReadVariableOp┌
0face_net/batch_normalization_60/ReadVariableOp_1ReadVariableOp9face_net_batch_normalization_60_readvariableop_1_resource*
_output_shapes
:*
dtype022
0face_net/batch_normalization_60/ReadVariableOp_1З
?face_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOpReadVariableOpHface_net_batch_normalization_60_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02A
?face_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOpН
Aface_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJface_net_batch_normalization_60_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02C
Aface_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1п
0face_net/batch_normalization_60/FusedBatchNormV3FusedBatchNormV3)face_net/activation_60/Relu:activations:06face_net/batch_normalization_60/ReadVariableOp:value:08face_net/batch_normalization_60/ReadVariableOp_1:value:0Gface_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOp:value:0Iface_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
is_training( 22
0face_net/batch_normalization_60/FusedBatchNormV3Ї
!face_net/max_pooling2d_48/MaxPoolMaxPool4face_net/batch_normalization_64/FusedBatchNormV3:y:0*/
_output_shapes
:         BB*
ksize
*
paddingVALID*
strides
2#
!face_net/max_pooling2d_48/MaxPoolЇ
!face_net/max_pooling2d_45/MaxPoolMaxPool4face_net/batch_normalization_60/FusedBatchNormV3:y:0*/
_output_shapes
:         BB*
ksize
*
paddingVALID*
strides
2#
!face_net/max_pooling2d_45/MaxPoolо
face_net/dropout_64/IdentityIdentity*face_net/max_pooling2d_48/MaxPool:output:0*
T0*/
_output_shapes
:         BB2
face_net/dropout_64/Identityо
face_net/dropout_60/IdentityIdentity*face_net/max_pooling2d_45/MaxPool:output:0*
T0*/
_output_shapes
:         BB2
face_net/dropout_60/Identity╬
(face_net/conv2d_49/Conv2D/ReadVariableOpReadVariableOp1face_net_conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(face_net/conv2d_49/Conv2D/ReadVariableOp√
face_net/conv2d_49/Conv2DConv2D%face_net/dropout_64/Identity:output:00face_net/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB *
paddingSAME*
strides
2
face_net/conv2d_49/Conv2D┼
)face_net/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp2face_net_conv2d_49_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)face_net/conv2d_49/BiasAdd/ReadVariableOp╘
face_net/conv2d_49/BiasAddBiasAdd"face_net/conv2d_49/Conv2D:output:01face_net/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB 2
face_net/conv2d_49/BiasAdd╬
(face_net/conv2d_46/Conv2D/ReadVariableOpReadVariableOp1face_net_conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(face_net/conv2d_46/Conv2D/ReadVariableOp√
face_net/conv2d_46/Conv2DConv2D%face_net/dropout_60/Identity:output:00face_net/conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB *
paddingSAME*
strides
2
face_net/conv2d_46/Conv2D┼
)face_net/conv2d_46/BiasAdd/ReadVariableOpReadVariableOp2face_net_conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)face_net/conv2d_46/BiasAdd/ReadVariableOp╘
face_net/conv2d_46/BiasAddBiasAdd"face_net/conv2d_46/Conv2D:output:01face_net/conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB 2
face_net/conv2d_46/BiasAddб
face_net/activation_65/ReluRelu#face_net/conv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:         BB 2
face_net/activation_65/Reluб
face_net/activation_61/ReluRelu#face_net/conv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:         BB 2
face_net/activation_61/Relu╘
.face_net/batch_normalization_65/ReadVariableOpReadVariableOp7face_net_batch_normalization_65_readvariableop_resource*
_output_shapes
: *
dtype020
.face_net/batch_normalization_65/ReadVariableOp┌
0face_net/batch_normalization_65/ReadVariableOp_1ReadVariableOp9face_net_batch_normalization_65_readvariableop_1_resource*
_output_shapes
: *
dtype022
0face_net/batch_normalization_65/ReadVariableOp_1З
?face_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOpReadVariableOpHface_net_batch_normalization_65_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02A
?face_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOpН
Aface_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJface_net_batch_normalization_65_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02C
Aface_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1н
0face_net/batch_normalization_65/FusedBatchNormV3FusedBatchNormV3)face_net/activation_65/Relu:activations:06face_net/batch_normalization_65/ReadVariableOp:value:08face_net/batch_normalization_65/ReadVariableOp_1:value:0Gface_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOp:value:0Iface_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
is_training( 22
0face_net/batch_normalization_65/FusedBatchNormV3╘
.face_net/batch_normalization_61/ReadVariableOpReadVariableOp7face_net_batch_normalization_61_readvariableop_resource*
_output_shapes
: *
dtype020
.face_net/batch_normalization_61/ReadVariableOp┌
0face_net/batch_normalization_61/ReadVariableOp_1ReadVariableOp9face_net_batch_normalization_61_readvariableop_1_resource*
_output_shapes
: *
dtype022
0face_net/batch_normalization_61/ReadVariableOp_1З
?face_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOpReadVariableOpHface_net_batch_normalization_61_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02A
?face_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOpН
Aface_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJface_net_batch_normalization_61_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02C
Aface_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1н
0face_net/batch_normalization_61/FusedBatchNormV3FusedBatchNormV3)face_net/activation_61/Relu:activations:06face_net/batch_normalization_61/ReadVariableOp:value:08face_net/batch_normalization_61/ReadVariableOp_1:value:0Gface_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOp:value:0Iface_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
is_training( 22
0face_net/batch_normalization_61/FusedBatchNormV3Ї
!face_net/max_pooling2d_49/MaxPoolMaxPool4face_net/batch_normalization_65/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2#
!face_net/max_pooling2d_49/MaxPoolЇ
!face_net/max_pooling2d_46/MaxPoolMaxPool4face_net/batch_normalization_61/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2#
!face_net/max_pooling2d_46/MaxPoolо
face_net/dropout_65/IdentityIdentity*face_net/max_pooling2d_49/MaxPool:output:0*
T0*/
_output_shapes
:         !! 2
face_net/dropout_65/Identityо
face_net/dropout_61/IdentityIdentity*face_net/max_pooling2d_46/MaxPool:output:0*
T0*/
_output_shapes
:         !! 2
face_net/dropout_61/Identity╬
(face_net/conv2d_50/Conv2D/ReadVariableOpReadVariableOp1face_net_conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02*
(face_net/conv2d_50/Conv2D/ReadVariableOp√
face_net/conv2d_50/Conv2DConv2D%face_net/dropout_65/Identity:output:00face_net/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! *
paddingSAME*
strides
2
face_net/conv2d_50/Conv2D┼
)face_net/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp2face_net_conv2d_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)face_net/conv2d_50/BiasAdd/ReadVariableOp╘
face_net/conv2d_50/BiasAddBiasAdd"face_net/conv2d_50/Conv2D:output:01face_net/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! 2
face_net/conv2d_50/BiasAdd╬
(face_net/conv2d_47/Conv2D/ReadVariableOpReadVariableOp1face_net_conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02*
(face_net/conv2d_47/Conv2D/ReadVariableOp√
face_net/conv2d_47/Conv2DConv2D%face_net/dropout_61/Identity:output:00face_net/conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! *
paddingSAME*
strides
2
face_net/conv2d_47/Conv2D┼
)face_net/conv2d_47/BiasAdd/ReadVariableOpReadVariableOp2face_net_conv2d_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)face_net/conv2d_47/BiasAdd/ReadVariableOp╘
face_net/conv2d_47/BiasAddBiasAdd"face_net/conv2d_47/Conv2D:output:01face_net/conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! 2
face_net/conv2d_47/BiasAddб
face_net/activation_66/ReluRelu#face_net/conv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:         !! 2
face_net/activation_66/Reluб
face_net/activation_62/ReluRelu#face_net/conv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:         !! 2
face_net/activation_62/Relu╘
.face_net/batch_normalization_66/ReadVariableOpReadVariableOp7face_net_batch_normalization_66_readvariableop_resource*
_output_shapes
: *
dtype020
.face_net/batch_normalization_66/ReadVariableOp┌
0face_net/batch_normalization_66/ReadVariableOp_1ReadVariableOp9face_net_batch_normalization_66_readvariableop_1_resource*
_output_shapes
: *
dtype022
0face_net/batch_normalization_66/ReadVariableOp_1З
?face_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOpReadVariableOpHface_net_batch_normalization_66_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02A
?face_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOpН
Aface_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJface_net_batch_normalization_66_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02C
Aface_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1н
0face_net/batch_normalization_66/FusedBatchNormV3FusedBatchNormV3)face_net/activation_66/Relu:activations:06face_net/batch_normalization_66/ReadVariableOp:value:08face_net/batch_normalization_66/ReadVariableOp_1:value:0Gface_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOp:value:0Iface_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 22
0face_net/batch_normalization_66/FusedBatchNormV3╘
.face_net/batch_normalization_62/ReadVariableOpReadVariableOp7face_net_batch_normalization_62_readvariableop_resource*
_output_shapes
: *
dtype020
.face_net/batch_normalization_62/ReadVariableOp┌
0face_net/batch_normalization_62/ReadVariableOp_1ReadVariableOp9face_net_batch_normalization_62_readvariableop_1_resource*
_output_shapes
: *
dtype022
0face_net/batch_normalization_62/ReadVariableOp_1З
?face_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOpReadVariableOpHface_net_batch_normalization_62_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02A
?face_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOpН
Aface_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJface_net_batch_normalization_62_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02C
Aface_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1н
0face_net/batch_normalization_62/FusedBatchNormV3FusedBatchNormV3)face_net/activation_62/Relu:activations:06face_net/batch_normalization_62/ReadVariableOp:value:08face_net/batch_normalization_62/ReadVariableOp_1:value:0Gface_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOp:value:0Iface_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 22
0face_net/batch_normalization_62/FusedBatchNormV3Ї
!face_net/max_pooling2d_50/MaxPoolMaxPool4face_net/batch_normalization_66/FusedBatchNormV3:y:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2#
!face_net/max_pooling2d_50/MaxPoolЇ
!face_net/max_pooling2d_47/MaxPoolMaxPool4face_net/batch_normalization_62/FusedBatchNormV3:y:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2#
!face_net/max_pooling2d_47/MaxPoolо
face_net/dropout_66/IdentityIdentity*face_net/max_pooling2d_50/MaxPool:output:0*
T0*/
_output_shapes
:          2
face_net/dropout_66/Identityо
face_net/dropout_62/IdentityIdentity*face_net/max_pooling2d_47/MaxPool:output:0*
T0*/
_output_shapes
:          2
face_net/dropout_62/IdentityЗ
face_net/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
face_net/flatten_16/Const├
face_net/flatten_16/ReshapeReshape%face_net/dropout_66/Identity:output:0"face_net/flatten_16/Const:output:0*
T0*(
_output_shapes
:         А@2
face_net/flatten_16/ReshapeЗ
face_net/flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
face_net/flatten_15/Const├
face_net/flatten_15/ReshapeReshape%face_net/dropout_62/Identity:output:0"face_net/flatten_15/Const:output:0*
T0*(
_output_shapes
:         А@2
face_net/flatten_15/Reshape┼
'face_net/dense_32/MatMul/ReadVariableOpReadVariableOp0face_net_dense_32_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02)
'face_net/dense_32/MatMul/ReadVariableOp╚
face_net/dense_32/MatMulMatMul$face_net/flatten_16/Reshape:output:0/face_net/dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
face_net/dense_32/MatMul├
(face_net/dense_32/BiasAdd/ReadVariableOpReadVariableOp1face_net_dense_32_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(face_net/dense_32/BiasAdd/ReadVariableOp╩
face_net/dense_32/BiasAddBiasAdd"face_net/dense_32/MatMul:product:00face_net/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
face_net/dense_32/BiasAdd┼
'face_net/dense_30/MatMul/ReadVariableOpReadVariableOp0face_net_dense_30_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02)
'face_net/dense_30/MatMul/ReadVariableOp╚
face_net/dense_30/MatMulMatMul$face_net/flatten_15/Reshape:output:0/face_net/dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
face_net/dense_30/MatMul├
(face_net/dense_30/BiasAdd/ReadVariableOpReadVariableOp1face_net_dense_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(face_net/dense_30/BiasAdd/ReadVariableOp╩
face_net/dense_30/BiasAddBiasAdd"face_net/dense_30/MatMul:product:00face_net/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
face_net/dense_30/BiasAddЩ
face_net/activation_67/ReluRelu"face_net/dense_32/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
face_net/activation_67/ReluЩ
face_net/activation_63/ReluRelu"face_net/dense_30/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
face_net/activation_63/Reluє
8face_net/batch_normalization_67/batchnorm/ReadVariableOpReadVariableOpAface_net_batch_normalization_67_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02:
8face_net/batch_normalization_67/batchnorm/ReadVariableOpз
/face_net/batch_normalization_67/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:21
/face_net/batch_normalization_67/batchnorm/add/yЙ
-face_net/batch_normalization_67/batchnorm/addAddV2@face_net/batch_normalization_67/batchnorm/ReadVariableOp:value:08face_net/batch_normalization_67/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2/
-face_net/batch_normalization_67/batchnorm/add─
/face_net/batch_normalization_67/batchnorm/RsqrtRsqrt1face_net/batch_normalization_67/batchnorm/add:z:0*
T0*
_output_shapes	
:А21
/face_net/batch_normalization_67/batchnorm/Rsqrt 
<face_net/batch_normalization_67/batchnorm/mul/ReadVariableOpReadVariableOpEface_net_batch_normalization_67_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02>
<face_net/batch_normalization_67/batchnorm/mul/ReadVariableOpЖ
-face_net/batch_normalization_67/batchnorm/mulMul3face_net/batch_normalization_67/batchnorm/Rsqrt:y:0Dface_net/batch_normalization_67/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2/
-face_net/batch_normalization_67/batchnorm/mul·
/face_net/batch_normalization_67/batchnorm/mul_1Mul)face_net/activation_67/Relu:activations:01face_net/batch_normalization_67/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А21
/face_net/batch_normalization_67/batchnorm/mul_1∙
:face_net/batch_normalization_67/batchnorm/ReadVariableOp_1ReadVariableOpCface_net_batch_normalization_67_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02<
:face_net/batch_normalization_67/batchnorm/ReadVariableOp_1Ж
/face_net/batch_normalization_67/batchnorm/mul_2MulBface_net/batch_normalization_67/batchnorm/ReadVariableOp_1:value:01face_net/batch_normalization_67/batchnorm/mul:z:0*
T0*
_output_shapes	
:А21
/face_net/batch_normalization_67/batchnorm/mul_2∙
:face_net/batch_normalization_67/batchnorm/ReadVariableOp_2ReadVariableOpCface_net_batch_normalization_67_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02<
:face_net/batch_normalization_67/batchnorm/ReadVariableOp_2Д
-face_net/batch_normalization_67/batchnorm/subSubBface_net/batch_normalization_67/batchnorm/ReadVariableOp_2:value:03face_net/batch_normalization_67/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2/
-face_net/batch_normalization_67/batchnorm/subЖ
/face_net/batch_normalization_67/batchnorm/add_1AddV23face_net/batch_normalization_67/batchnorm/mul_1:z:01face_net/batch_normalization_67/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А21
/face_net/batch_normalization_67/batchnorm/add_1є
8face_net/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOpAface_net_batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02:
8face_net/batch_normalization_63/batchnorm/ReadVariableOpз
/face_net/batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:21
/face_net/batch_normalization_63/batchnorm/add/yЙ
-face_net/batch_normalization_63/batchnorm/addAddV2@face_net/batch_normalization_63/batchnorm/ReadVariableOp:value:08face_net/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2/
-face_net/batch_normalization_63/batchnorm/add─
/face_net/batch_normalization_63/batchnorm/RsqrtRsqrt1face_net/batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes	
:А21
/face_net/batch_normalization_63/batchnorm/Rsqrt 
<face_net/batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOpEface_net_batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02>
<face_net/batch_normalization_63/batchnorm/mul/ReadVariableOpЖ
-face_net/batch_normalization_63/batchnorm/mulMul3face_net/batch_normalization_63/batchnorm/Rsqrt:y:0Dface_net/batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2/
-face_net/batch_normalization_63/batchnorm/mul·
/face_net/batch_normalization_63/batchnorm/mul_1Mul)face_net/activation_63/Relu:activations:01face_net/batch_normalization_63/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А21
/face_net/batch_normalization_63/batchnorm/mul_1∙
:face_net/batch_normalization_63/batchnorm/ReadVariableOp_1ReadVariableOpCface_net_batch_normalization_63_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02<
:face_net/batch_normalization_63/batchnorm/ReadVariableOp_1Ж
/face_net/batch_normalization_63/batchnorm/mul_2MulBface_net/batch_normalization_63/batchnorm/ReadVariableOp_1:value:01face_net/batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes	
:А21
/face_net/batch_normalization_63/batchnorm/mul_2∙
:face_net/batch_normalization_63/batchnorm/ReadVariableOp_2ReadVariableOpCface_net_batch_normalization_63_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02<
:face_net/batch_normalization_63/batchnorm/ReadVariableOp_2Д
-face_net/batch_normalization_63/batchnorm/subSubBface_net/batch_normalization_63/batchnorm/ReadVariableOp_2:value:03face_net/batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2/
-face_net/batch_normalization_63/batchnorm/subЖ
/face_net/batch_normalization_63/batchnorm/add_1AddV23face_net/batch_normalization_63/batchnorm/mul_1:z:01face_net/batch_normalization_63/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А21
/face_net/batch_normalization_63/batchnorm/add_1░
face_net/dropout_67/IdentityIdentity3face_net/batch_normalization_67/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         А2
face_net/dropout_67/Identity░
face_net/dropout_63/IdentityIdentity3face_net/batch_normalization_63/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         А2
face_net/dropout_63/Identity─
'face_net/dense_33/MatMul/ReadVariableOpReadVariableOp0face_net_dense_33_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02)
'face_net/dense_33/MatMul/ReadVariableOp╚
face_net/dense_33/MatMulMatMul%face_net/dropout_67/Identity:output:0/face_net/dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
face_net/dense_33/MatMul┬
(face_net/dense_33/BiasAdd/ReadVariableOpReadVariableOp1face_net_dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(face_net/dense_33/BiasAdd/ReadVariableOp╔
face_net/dense_33/BiasAddBiasAdd"face_net/dense_33/MatMul:product:00face_net/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
face_net/dense_33/BiasAdd─
'face_net/dense_31/MatMul/ReadVariableOpReadVariableOp0face_net_dense_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02)
'face_net/dense_31/MatMul/ReadVariableOp╚
face_net/dense_31/MatMulMatMul%face_net/dropout_63/Identity:output:0/face_net/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
face_net/dense_31/MatMul┬
(face_net/dense_31/BiasAdd/ReadVariableOpReadVariableOp1face_net_dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(face_net/dense_31/BiasAdd/ReadVariableOp╔
face_net/dense_31/BiasAddBiasAdd"face_net/dense_31/MatMul:product:00face_net/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
face_net/dense_31/BiasAddб
face_net/gender_output/SigmoidSigmoid"face_net/dense_33/BiasAdd:output:0*
T0*'
_output_shapes
:         2 
face_net/gender_output/Sigmoid╕
IdentityIdentity"face_net/dense_31/BiasAdd:output:0@^face_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_60/ReadVariableOp1^face_net/batch_normalization_60/ReadVariableOp_1@^face_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_61/ReadVariableOp1^face_net/batch_normalization_61/ReadVariableOp_1@^face_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_62/ReadVariableOp1^face_net/batch_normalization_62/ReadVariableOp_19^face_net/batch_normalization_63/batchnorm/ReadVariableOp;^face_net/batch_normalization_63/batchnorm/ReadVariableOp_1;^face_net/batch_normalization_63/batchnorm/ReadVariableOp_2=^face_net/batch_normalization_63/batchnorm/mul/ReadVariableOp@^face_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_64/ReadVariableOp1^face_net/batch_normalization_64/ReadVariableOp_1@^face_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_65/ReadVariableOp1^face_net/batch_normalization_65/ReadVariableOp_1@^face_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_66/ReadVariableOp1^face_net/batch_normalization_66/ReadVariableOp_19^face_net/batch_normalization_67/batchnorm/ReadVariableOp;^face_net/batch_normalization_67/batchnorm/ReadVariableOp_1;^face_net/batch_normalization_67/batchnorm/ReadVariableOp_2=^face_net/batch_normalization_67/batchnorm/mul/ReadVariableOp*^face_net/conv2d_45/BiasAdd/ReadVariableOp)^face_net/conv2d_45/Conv2D/ReadVariableOp*^face_net/conv2d_46/BiasAdd/ReadVariableOp)^face_net/conv2d_46/Conv2D/ReadVariableOp*^face_net/conv2d_47/BiasAdd/ReadVariableOp)^face_net/conv2d_47/Conv2D/ReadVariableOp*^face_net/conv2d_48/BiasAdd/ReadVariableOp)^face_net/conv2d_48/Conv2D/ReadVariableOp*^face_net/conv2d_49/BiasAdd/ReadVariableOp)^face_net/conv2d_49/Conv2D/ReadVariableOp*^face_net/conv2d_50/BiasAdd/ReadVariableOp)^face_net/conv2d_50/Conv2D/ReadVariableOp)^face_net/dense_30/BiasAdd/ReadVariableOp(^face_net/dense_30/MatMul/ReadVariableOp)^face_net/dense_31/BiasAdd/ReadVariableOp(^face_net/dense_31/MatMul/ReadVariableOp)^face_net/dense_32/BiasAdd/ReadVariableOp(^face_net/dense_32/MatMul/ReadVariableOp)^face_net/dense_33/BiasAdd/ReadVariableOp(^face_net/dense_33/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity╝

Identity_1Identity"face_net/gender_output/Sigmoid:y:0@^face_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_60/ReadVariableOp1^face_net/batch_normalization_60/ReadVariableOp_1@^face_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_61/ReadVariableOp1^face_net/batch_normalization_61/ReadVariableOp_1@^face_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_62/ReadVariableOp1^face_net/batch_normalization_62/ReadVariableOp_19^face_net/batch_normalization_63/batchnorm/ReadVariableOp;^face_net/batch_normalization_63/batchnorm/ReadVariableOp_1;^face_net/batch_normalization_63/batchnorm/ReadVariableOp_2=^face_net/batch_normalization_63/batchnorm/mul/ReadVariableOp@^face_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_64/ReadVariableOp1^face_net/batch_normalization_64/ReadVariableOp_1@^face_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_65/ReadVariableOp1^face_net/batch_normalization_65/ReadVariableOp_1@^face_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOpB^face_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1/^face_net/batch_normalization_66/ReadVariableOp1^face_net/batch_normalization_66/ReadVariableOp_19^face_net/batch_normalization_67/batchnorm/ReadVariableOp;^face_net/batch_normalization_67/batchnorm/ReadVariableOp_1;^face_net/batch_normalization_67/batchnorm/ReadVariableOp_2=^face_net/batch_normalization_67/batchnorm/mul/ReadVariableOp*^face_net/conv2d_45/BiasAdd/ReadVariableOp)^face_net/conv2d_45/Conv2D/ReadVariableOp*^face_net/conv2d_46/BiasAdd/ReadVariableOp)^face_net/conv2d_46/Conv2D/ReadVariableOp*^face_net/conv2d_47/BiasAdd/ReadVariableOp)^face_net/conv2d_47/Conv2D/ReadVariableOp*^face_net/conv2d_48/BiasAdd/ReadVariableOp)^face_net/conv2d_48/Conv2D/ReadVariableOp*^face_net/conv2d_49/BiasAdd/ReadVariableOp)^face_net/conv2d_49/Conv2D/ReadVariableOp*^face_net/conv2d_50/BiasAdd/ReadVariableOp)^face_net/conv2d_50/Conv2D/ReadVariableOp)^face_net/dense_30/BiasAdd/ReadVariableOp(^face_net/dense_30/MatMul/ReadVariableOp)^face_net/dense_31/BiasAdd/ReadVariableOp(^face_net/dense_31/MatMul/ReadVariableOp)^face_net/dense_32/BiasAdd/ReadVariableOp(^face_net/dense_32/MatMul/ReadVariableOp)^face_net/dense_33/BiasAdd/ReadVariableOp(^face_net/dense_33/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::2В
?face_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOp?face_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOp2Ж
Aface_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1Aface_net/batch_normalization_60/FusedBatchNormV3/ReadVariableOp_12`
.face_net/batch_normalization_60/ReadVariableOp.face_net/batch_normalization_60/ReadVariableOp2d
0face_net/batch_normalization_60/ReadVariableOp_10face_net/batch_normalization_60/ReadVariableOp_12В
?face_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOp?face_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOp2Ж
Aface_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1Aface_net/batch_normalization_61/FusedBatchNormV3/ReadVariableOp_12`
.face_net/batch_normalization_61/ReadVariableOp.face_net/batch_normalization_61/ReadVariableOp2d
0face_net/batch_normalization_61/ReadVariableOp_10face_net/batch_normalization_61/ReadVariableOp_12В
?face_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOp?face_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOp2Ж
Aface_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1Aface_net/batch_normalization_62/FusedBatchNormV3/ReadVariableOp_12`
.face_net/batch_normalization_62/ReadVariableOp.face_net/batch_normalization_62/ReadVariableOp2d
0face_net/batch_normalization_62/ReadVariableOp_10face_net/batch_normalization_62/ReadVariableOp_12t
8face_net/batch_normalization_63/batchnorm/ReadVariableOp8face_net/batch_normalization_63/batchnorm/ReadVariableOp2x
:face_net/batch_normalization_63/batchnorm/ReadVariableOp_1:face_net/batch_normalization_63/batchnorm/ReadVariableOp_12x
:face_net/batch_normalization_63/batchnorm/ReadVariableOp_2:face_net/batch_normalization_63/batchnorm/ReadVariableOp_22|
<face_net/batch_normalization_63/batchnorm/mul/ReadVariableOp<face_net/batch_normalization_63/batchnorm/mul/ReadVariableOp2В
?face_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOp?face_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOp2Ж
Aface_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1Aface_net/batch_normalization_64/FusedBatchNormV3/ReadVariableOp_12`
.face_net/batch_normalization_64/ReadVariableOp.face_net/batch_normalization_64/ReadVariableOp2d
0face_net/batch_normalization_64/ReadVariableOp_10face_net/batch_normalization_64/ReadVariableOp_12В
?face_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOp?face_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOp2Ж
Aface_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1Aface_net/batch_normalization_65/FusedBatchNormV3/ReadVariableOp_12`
.face_net/batch_normalization_65/ReadVariableOp.face_net/batch_normalization_65/ReadVariableOp2d
0face_net/batch_normalization_65/ReadVariableOp_10face_net/batch_normalization_65/ReadVariableOp_12В
?face_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOp?face_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOp2Ж
Aface_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1Aface_net/batch_normalization_66/FusedBatchNormV3/ReadVariableOp_12`
.face_net/batch_normalization_66/ReadVariableOp.face_net/batch_normalization_66/ReadVariableOp2d
0face_net/batch_normalization_66/ReadVariableOp_10face_net/batch_normalization_66/ReadVariableOp_12t
8face_net/batch_normalization_67/batchnorm/ReadVariableOp8face_net/batch_normalization_67/batchnorm/ReadVariableOp2x
:face_net/batch_normalization_67/batchnorm/ReadVariableOp_1:face_net/batch_normalization_67/batchnorm/ReadVariableOp_12x
:face_net/batch_normalization_67/batchnorm/ReadVariableOp_2:face_net/batch_normalization_67/batchnorm/ReadVariableOp_22|
<face_net/batch_normalization_67/batchnorm/mul/ReadVariableOp<face_net/batch_normalization_67/batchnorm/mul/ReadVariableOp2V
)face_net/conv2d_45/BiasAdd/ReadVariableOp)face_net/conv2d_45/BiasAdd/ReadVariableOp2T
(face_net/conv2d_45/Conv2D/ReadVariableOp(face_net/conv2d_45/Conv2D/ReadVariableOp2V
)face_net/conv2d_46/BiasAdd/ReadVariableOp)face_net/conv2d_46/BiasAdd/ReadVariableOp2T
(face_net/conv2d_46/Conv2D/ReadVariableOp(face_net/conv2d_46/Conv2D/ReadVariableOp2V
)face_net/conv2d_47/BiasAdd/ReadVariableOp)face_net/conv2d_47/BiasAdd/ReadVariableOp2T
(face_net/conv2d_47/Conv2D/ReadVariableOp(face_net/conv2d_47/Conv2D/ReadVariableOp2V
)face_net/conv2d_48/BiasAdd/ReadVariableOp)face_net/conv2d_48/BiasAdd/ReadVariableOp2T
(face_net/conv2d_48/Conv2D/ReadVariableOp(face_net/conv2d_48/Conv2D/ReadVariableOp2V
)face_net/conv2d_49/BiasAdd/ReadVariableOp)face_net/conv2d_49/BiasAdd/ReadVariableOp2T
(face_net/conv2d_49/Conv2D/ReadVariableOp(face_net/conv2d_49/Conv2D/ReadVariableOp2V
)face_net/conv2d_50/BiasAdd/ReadVariableOp)face_net/conv2d_50/BiasAdd/ReadVariableOp2T
(face_net/conv2d_50/Conv2D/ReadVariableOp(face_net/conv2d_50/Conv2D/ReadVariableOp2T
(face_net/dense_30/BiasAdd/ReadVariableOp(face_net/dense_30/BiasAdd/ReadVariableOp2R
'face_net/dense_30/MatMul/ReadVariableOp'face_net/dense_30/MatMul/ReadVariableOp2T
(face_net/dense_31/BiasAdd/ReadVariableOp(face_net/dense_31/BiasAdd/ReadVariableOp2R
'face_net/dense_31/MatMul/ReadVariableOp'face_net/dense_31/MatMul/ReadVariableOp2T
(face_net/dense_32/BiasAdd/ReadVariableOp(face_net/dense_32/BiasAdd/ReadVariableOp2R
'face_net/dense_32/MatMul/ReadVariableOp'face_net/dense_32/MatMul/ReadVariableOp2T
(face_net/dense_33/BiasAdd/ReadVariableOp(face_net/dense_33/BiasAdd/ReadVariableOp2R
'face_net/dense_33/MatMul/ReadVariableOp'face_net/dense_33/MatMul/ReadVariableOp:Z V
1
_output_shapes
:         ╞╞
!
_user_specified_name	input_6
С0
╞
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_38630

inputs
assignmovingavg_38605
assignmovingavg_1_38611)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/38605*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_38605*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/38605*
_output_shapes	
:А2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/38605*
_output_shapes	
:А2
AssignMovingAvg/mulн
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_38605AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/38605*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╤
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/38611*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_38611*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp√
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/38611*
_output_shapes	
:А2
AssignMovingAvg_1/subЄ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/38611*
_output_shapes	
:А2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_38611AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/38611*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
├
c
*__inference_dropout_62_layer_call_fn_42604

inputs
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_62_layer_call_and_return_conditional_losses_396582
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
├
c
*__inference_dropout_60_layer_call_fn_41868

inputs
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_60_layer_call_and_return_conditional_losses_390862
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         BB2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
├
d
E__inference_dropout_60_layer_call_and_return_conditional_losses_39086

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         BB2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         BB*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         BB2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         BB2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         BB2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         BB2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
▐
d
H__inference_activation_64_layer_call_and_return_conditional_losses_41585

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:         ╞╞2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ╞╞:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
╫
a
E__inference_age_output_layer_call_and_return_conditional_losses_39981

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
а
й
6__inference_batch_normalization_66_layer_call_fn_42505

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_384682
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
░
L
0__inference_max_pooling2d_48_layer_call_fn_38070

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_380642
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Н
Ш
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41738

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1А
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
ХК
Я?
__inference__traced_save_43432
file_prefix/
+savev2_conv2d_45_kernel_read_readvariableop-
)savev2_conv2d_45_bias_read_readvariableop/
+savev2_conv2d_48_kernel_read_readvariableop-
)savev2_conv2d_48_bias_read_readvariableop;
7savev2_batch_normalization_60_gamma_read_readvariableop:
6savev2_batch_normalization_60_beta_read_readvariableopA
=savev2_batch_normalization_60_moving_mean_read_readvariableopE
Asavev2_batch_normalization_60_moving_variance_read_readvariableop;
7savev2_batch_normalization_64_gamma_read_readvariableop:
6savev2_batch_normalization_64_beta_read_readvariableopA
=savev2_batch_normalization_64_moving_mean_read_readvariableopE
Asavev2_batch_normalization_64_moving_variance_read_readvariableop/
+savev2_conv2d_46_kernel_read_readvariableop-
)savev2_conv2d_46_bias_read_readvariableop/
+savev2_conv2d_49_kernel_read_readvariableop-
)savev2_conv2d_49_bias_read_readvariableop;
7savev2_batch_normalization_61_gamma_read_readvariableop:
6savev2_batch_normalization_61_beta_read_readvariableopA
=savev2_batch_normalization_61_moving_mean_read_readvariableopE
Asavev2_batch_normalization_61_moving_variance_read_readvariableop;
7savev2_batch_normalization_65_gamma_read_readvariableop:
6savev2_batch_normalization_65_beta_read_readvariableopA
=savev2_batch_normalization_65_moving_mean_read_readvariableopE
Asavev2_batch_normalization_65_moving_variance_read_readvariableop/
+savev2_conv2d_47_kernel_read_readvariableop-
)savev2_conv2d_47_bias_read_readvariableop/
+savev2_conv2d_50_kernel_read_readvariableop-
)savev2_conv2d_50_bias_read_readvariableop;
7savev2_batch_normalization_62_gamma_read_readvariableop:
6savev2_batch_normalization_62_beta_read_readvariableopA
=savev2_batch_normalization_62_moving_mean_read_readvariableopE
Asavev2_batch_normalization_62_moving_variance_read_readvariableop;
7savev2_batch_normalization_66_gamma_read_readvariableop:
6savev2_batch_normalization_66_beta_read_readvariableopA
=savev2_batch_normalization_66_moving_mean_read_readvariableopE
Asavev2_batch_normalization_66_moving_variance_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop;
7savev2_batch_normalization_63_gamma_read_readvariableop:
6savev2_batch_normalization_63_beta_read_readvariableopA
=savev2_batch_normalization_63_moving_mean_read_readvariableopE
Asavev2_batch_normalization_63_moving_variance_read_readvariableop;
7savev2_batch_normalization_67_gamma_read_readvariableop:
6savev2_batch_normalization_67_beta_read_readvariableopA
=savev2_batch_normalization_67_moving_mean_read_readvariableopE
Asavev2_batch_normalization_67_moving_variance_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop6
2savev2_adam_conv2d_45_kernel_m_read_readvariableop4
0savev2_adam_conv2d_45_bias_m_read_readvariableop6
2savev2_adam_conv2d_48_kernel_m_read_readvariableop4
0savev2_adam_conv2d_48_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_60_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_60_beta_m_read_readvariableopB
>savev2_adam_batch_normalization_64_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_64_beta_m_read_readvariableop6
2savev2_adam_conv2d_46_kernel_m_read_readvariableop4
0savev2_adam_conv2d_46_bias_m_read_readvariableop6
2savev2_adam_conv2d_49_kernel_m_read_readvariableop4
0savev2_adam_conv2d_49_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_61_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_61_beta_m_read_readvariableopB
>savev2_adam_batch_normalization_65_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_65_beta_m_read_readvariableop6
2savev2_adam_conv2d_47_kernel_m_read_readvariableop4
0savev2_adam_conv2d_47_bias_m_read_readvariableop6
2savev2_adam_conv2d_50_kernel_m_read_readvariableop4
0savev2_adam_conv2d_50_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_62_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_62_beta_m_read_readvariableopB
>savev2_adam_batch_normalization_66_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_66_beta_m_read_readvariableop5
1savev2_adam_dense_30_kernel_m_read_readvariableop3
/savev2_adam_dense_30_bias_m_read_readvariableop5
1savev2_adam_dense_32_kernel_m_read_readvariableop3
/savev2_adam_dense_32_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_63_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_63_beta_m_read_readvariableopB
>savev2_adam_batch_normalization_67_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_67_beta_m_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableop5
1savev2_adam_dense_33_kernel_m_read_readvariableop3
/savev2_adam_dense_33_bias_m_read_readvariableop6
2savev2_adam_conv2d_45_kernel_v_read_readvariableop4
0savev2_adam_conv2d_45_bias_v_read_readvariableop6
2savev2_adam_conv2d_48_kernel_v_read_readvariableop4
0savev2_adam_conv2d_48_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_60_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_60_beta_v_read_readvariableopB
>savev2_adam_batch_normalization_64_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_64_beta_v_read_readvariableop6
2savev2_adam_conv2d_46_kernel_v_read_readvariableop4
0savev2_adam_conv2d_46_bias_v_read_readvariableop6
2savev2_adam_conv2d_49_kernel_v_read_readvariableop4
0savev2_adam_conv2d_49_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_61_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_61_beta_v_read_readvariableopB
>savev2_adam_batch_normalization_65_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_65_beta_v_read_readvariableop6
2savev2_adam_conv2d_47_kernel_v_read_readvariableop4
0savev2_adam_conv2d_47_bias_v_read_readvariableop6
2savev2_adam_conv2d_50_kernel_v_read_readvariableop4
0savev2_adam_conv2d_50_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_62_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_62_beta_v_read_readvariableopB
>savev2_adam_batch_normalization_66_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_66_beta_v_read_readvariableop5
1savev2_adam_dense_30_kernel_v_read_readvariableop3
/savev2_adam_dense_30_bias_v_read_readvariableop5
1savev2_adam_dense_32_kernel_v_read_readvariableop3
/savev2_adam_dense_32_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_63_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_63_beta_v_read_readvariableopB
>savev2_adam_batch_normalization_67_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_67_beta_v_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableop5
1savev2_adam_dense_33_kernel_v_read_readvariableop3
/savev2_adam_dense_33_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╚M
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:М*
dtype0*┘L
value╧LB╠LМB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesе
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:М*
dtype0*о
valueдBбМB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╤<
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_45_kernel_read_readvariableop)savev2_conv2d_45_bias_read_readvariableop+savev2_conv2d_48_kernel_read_readvariableop)savev2_conv2d_48_bias_read_readvariableop7savev2_batch_normalization_60_gamma_read_readvariableop6savev2_batch_normalization_60_beta_read_readvariableop=savev2_batch_normalization_60_moving_mean_read_readvariableopAsavev2_batch_normalization_60_moving_variance_read_readvariableop7savev2_batch_normalization_64_gamma_read_readvariableop6savev2_batch_normalization_64_beta_read_readvariableop=savev2_batch_normalization_64_moving_mean_read_readvariableopAsavev2_batch_normalization_64_moving_variance_read_readvariableop+savev2_conv2d_46_kernel_read_readvariableop)savev2_conv2d_46_bias_read_readvariableop+savev2_conv2d_49_kernel_read_readvariableop)savev2_conv2d_49_bias_read_readvariableop7savev2_batch_normalization_61_gamma_read_readvariableop6savev2_batch_normalization_61_beta_read_readvariableop=savev2_batch_normalization_61_moving_mean_read_readvariableopAsavev2_batch_normalization_61_moving_variance_read_readvariableop7savev2_batch_normalization_65_gamma_read_readvariableop6savev2_batch_normalization_65_beta_read_readvariableop=savev2_batch_normalization_65_moving_mean_read_readvariableopAsavev2_batch_normalization_65_moving_variance_read_readvariableop+savev2_conv2d_47_kernel_read_readvariableop)savev2_conv2d_47_bias_read_readvariableop+savev2_conv2d_50_kernel_read_readvariableop)savev2_conv2d_50_bias_read_readvariableop7savev2_batch_normalization_62_gamma_read_readvariableop6savev2_batch_normalization_62_beta_read_readvariableop=savev2_batch_normalization_62_moving_mean_read_readvariableopAsavev2_batch_normalization_62_moving_variance_read_readvariableop7savev2_batch_normalization_66_gamma_read_readvariableop6savev2_batch_normalization_66_beta_read_readvariableop=savev2_batch_normalization_66_moving_mean_read_readvariableopAsavev2_batch_normalization_66_moving_variance_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop7savev2_batch_normalization_63_gamma_read_readvariableop6savev2_batch_normalization_63_beta_read_readvariableop=savev2_batch_normalization_63_moving_mean_read_readvariableopAsavev2_batch_normalization_63_moving_variance_read_readvariableop7savev2_batch_normalization_67_gamma_read_readvariableop6savev2_batch_normalization_67_beta_read_readvariableop=savev2_batch_normalization_67_moving_mean_read_readvariableopAsavev2_batch_normalization_67_moving_variance_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop2savev2_adam_conv2d_45_kernel_m_read_readvariableop0savev2_adam_conv2d_45_bias_m_read_readvariableop2savev2_adam_conv2d_48_kernel_m_read_readvariableop0savev2_adam_conv2d_48_bias_m_read_readvariableop>savev2_adam_batch_normalization_60_gamma_m_read_readvariableop=savev2_adam_batch_normalization_60_beta_m_read_readvariableop>savev2_adam_batch_normalization_64_gamma_m_read_readvariableop=savev2_adam_batch_normalization_64_beta_m_read_readvariableop2savev2_adam_conv2d_46_kernel_m_read_readvariableop0savev2_adam_conv2d_46_bias_m_read_readvariableop2savev2_adam_conv2d_49_kernel_m_read_readvariableop0savev2_adam_conv2d_49_bias_m_read_readvariableop>savev2_adam_batch_normalization_61_gamma_m_read_readvariableop=savev2_adam_batch_normalization_61_beta_m_read_readvariableop>savev2_adam_batch_normalization_65_gamma_m_read_readvariableop=savev2_adam_batch_normalization_65_beta_m_read_readvariableop2savev2_adam_conv2d_47_kernel_m_read_readvariableop0savev2_adam_conv2d_47_bias_m_read_readvariableop2savev2_adam_conv2d_50_kernel_m_read_readvariableop0savev2_adam_conv2d_50_bias_m_read_readvariableop>savev2_adam_batch_normalization_62_gamma_m_read_readvariableop=savev2_adam_batch_normalization_62_beta_m_read_readvariableop>savev2_adam_batch_normalization_66_gamma_m_read_readvariableop=savev2_adam_batch_normalization_66_beta_m_read_readvariableop1savev2_adam_dense_30_kernel_m_read_readvariableop/savev2_adam_dense_30_bias_m_read_readvariableop1savev2_adam_dense_32_kernel_m_read_readvariableop/savev2_adam_dense_32_bias_m_read_readvariableop>savev2_adam_batch_normalization_63_gamma_m_read_readvariableop=savev2_adam_batch_normalization_63_beta_m_read_readvariableop>savev2_adam_batch_normalization_67_gamma_m_read_readvariableop=savev2_adam_batch_normalization_67_beta_m_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop1savev2_adam_dense_33_kernel_m_read_readvariableop/savev2_adam_dense_33_bias_m_read_readvariableop2savev2_adam_conv2d_45_kernel_v_read_readvariableop0savev2_adam_conv2d_45_bias_v_read_readvariableop2savev2_adam_conv2d_48_kernel_v_read_readvariableop0savev2_adam_conv2d_48_bias_v_read_readvariableop>savev2_adam_batch_normalization_60_gamma_v_read_readvariableop=savev2_adam_batch_normalization_60_beta_v_read_readvariableop>savev2_adam_batch_normalization_64_gamma_v_read_readvariableop=savev2_adam_batch_normalization_64_beta_v_read_readvariableop2savev2_adam_conv2d_46_kernel_v_read_readvariableop0savev2_adam_conv2d_46_bias_v_read_readvariableop2savev2_adam_conv2d_49_kernel_v_read_readvariableop0savev2_adam_conv2d_49_bias_v_read_readvariableop>savev2_adam_batch_normalization_61_gamma_v_read_readvariableop=savev2_adam_batch_normalization_61_beta_v_read_readvariableop>savev2_adam_batch_normalization_65_gamma_v_read_readvariableop=savev2_adam_batch_normalization_65_beta_v_read_readvariableop2savev2_adam_conv2d_47_kernel_v_read_readvariableop0savev2_adam_conv2d_47_bias_v_read_readvariableop2savev2_adam_conv2d_50_kernel_v_read_readvariableop0savev2_adam_conv2d_50_bias_v_read_readvariableop>savev2_adam_batch_normalization_62_gamma_v_read_readvariableop=savev2_adam_batch_normalization_62_beta_v_read_readvariableop>savev2_adam_batch_normalization_66_gamma_v_read_readvariableop=savev2_adam_batch_normalization_66_beta_v_read_readvariableop1savev2_adam_dense_30_kernel_v_read_readvariableop/savev2_adam_dense_30_bias_v_read_readvariableop1savev2_adam_dense_32_kernel_v_read_readvariableop/savev2_adam_dense_32_bias_v_read_readvariableop>savev2_adam_batch_normalization_63_gamma_v_read_readvariableop=savev2_adam_batch_normalization_63_beta_v_read_readvariableop>savev2_adam_batch_normalization_67_gamma_v_read_readvariableop=savev2_adam_batch_normalization_67_beta_v_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableop1savev2_adam_dense_33_kernel_v_read_readvariableop/savev2_adam_dense_33_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Э
dtypesТ
П2М	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*╧
_input_shapes╜
║: ::::::::::::: : : : : : : : : : : : :  : :  : : : : : : : : : :
А@А:А:
А@А:А:А:А:А:А:А:А:А:А:	А::	А:: : : : : : : : : : : : : : : ::::::::: : : : : : : : :  : :  : : : : : :
А@А:А:
А@А:А:А:А:А:А:	А::	А:::::::::: : : : : : : : :  : :  : : : : : :
А@А:А:
А@А:А:А:А:А:А:	А::	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :  

_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: :&%"
 
_output_shapes
:
А@А:!&

_output_shapes	
:А:&'"
 
_output_shapes
:
А@А:!(

_output_shapes	
:А:!)

_output_shapes	
:А:!*

_output_shapes	
:А:!+

_output_shapes	
:А:!,

_output_shapes	
:А:!-

_output_shapes	
:А:!.

_output_shapes	
:А:!/

_output_shapes	
:А:!0

_output_shapes	
:А:%1!

_output_shapes
:	А: 2

_output_shapes
::%3!

_output_shapes
:	А: 4

_output_shapes
::5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :,D(
&
_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
:: H

_output_shapes
:: I

_output_shapes
:: J

_output_shapes
:: K

_output_shapes
::,L(
&
_output_shapes
: : M

_output_shapes
: :,N(
&
_output_shapes
: : O

_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: : R

_output_shapes
: : S

_output_shapes
: :,T(
&
_output_shapes
:  : U

_output_shapes
: :,V(
&
_output_shapes
:  : W

_output_shapes
: : X

_output_shapes
: : Y

_output_shapes
: : Z

_output_shapes
: : [

_output_shapes
: :&\"
 
_output_shapes
:
А@А:!]

_output_shapes	
:А:&^"
 
_output_shapes
:
А@А:!_

_output_shapes	
:А:!`

_output_shapes	
:А:!a

_output_shapes	
:А:!b

_output_shapes	
:А:!c

_output_shapes	
:А:%d!

_output_shapes
:	А: e

_output_shapes
::%f!

_output_shapes
:	А: g

_output_shapes
::,h(
&
_output_shapes
:: i

_output_shapes
::,j(
&
_output_shapes
:: k

_output_shapes
:: l

_output_shapes
:: m

_output_shapes
:: n

_output_shapes
:: o

_output_shapes
::,p(
&
_output_shapes
: : q

_output_shapes
: :,r(
&
_output_shapes
: : s

_output_shapes
: : t

_output_shapes
: : u

_output_shapes
: : v

_output_shapes
: : w

_output_shapes
: :,x(
&
_output_shapes
:  : y

_output_shapes
: :,z(
&
_output_shapes
:  : {

_output_shapes
: : |

_output_shapes
: : }

_output_shapes
: : ~

_output_shapes
: : 

_output_shapes
: :'А"
 
_output_shapes
:
А@А:"Б

_output_shapes	
:А:'В"
 
_output_shapes
:
А@А:"Г

_output_shapes	
:А:"Д

_output_shapes	
:А:"Е

_output_shapes	
:А:"Ж

_output_shapes	
:А:"З

_output_shapes	
:А:&И!

_output_shapes
:	А:!Й

_output_shapes
::&К!

_output_shapes
:	А:!Л

_output_shapes
::М

_output_shapes
: 
▐
d
H__inference_activation_64_layer_call_and_return_conditional_losses_38875

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:         ╞╞2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ╞╞:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
╪
й
6__inference_batch_normalization_66_layer_call_fn_42569

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_394872
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
Ы
F
*__inference_dropout_67_layer_call_fn_42934

inputs
identity╟
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_67_layer_call_and_return_conditional_losses_398692
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╪
й
6__inference_batch_normalization_65_layer_call_fn_42201

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_392012
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
▀
}
(__inference_dense_31_layer_call_fn_42953

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_399482
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ш
c
E__inference_dropout_62_layer_call_and_return_conditional_losses_42599

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:          2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
■╓
у
C__inference_face_net_layer_call_and_return_conditional_losses_39991
input_6
conv2d_48_38839
conv2d_48_38841
conv2d_45_38865
conv2d_45_38867 
batch_normalization_64_38960 
batch_normalization_64_38962 
batch_normalization_64_38964 
batch_normalization_64_38966 
batch_normalization_60_39033 
batch_normalization_60_39035 
batch_normalization_60_39037 
batch_normalization_60_39039
conv2d_49_39125
conv2d_49_39127
conv2d_46_39151
conv2d_46_39153 
batch_normalization_65_39246 
batch_normalization_65_39248 
batch_normalization_65_39250 
batch_normalization_65_39252 
batch_normalization_61_39319 
batch_normalization_61_39321 
batch_normalization_61_39323 
batch_normalization_61_39325
conv2d_50_39411
conv2d_50_39413
conv2d_47_39437
conv2d_47_39439 
batch_normalization_66_39532 
batch_normalization_66_39534 
batch_normalization_66_39536 
batch_normalization_66_39538 
batch_normalization_62_39605 
batch_normalization_62_39607 
batch_normalization_62_39609 
batch_normalization_62_39611
dense_32_39725
dense_32_39727
dense_30_39751
dense_30_39753 
batch_normalization_67_39808 
batch_normalization_67_39810 
batch_normalization_67_39812 
batch_normalization_67_39814 
batch_normalization_63_39843 
batch_normalization_63_39845 
batch_normalization_63_39847 
batch_normalization_63_39849
dense_33_39933
dense_33_39935
dense_31_39959
dense_31_39961
identity

identity_1Ив.batch_normalization_60/StatefulPartitionedCallв.batch_normalization_61/StatefulPartitionedCallв.batch_normalization_62/StatefulPartitionedCallв.batch_normalization_63/StatefulPartitionedCallв.batch_normalization_64/StatefulPartitionedCallв.batch_normalization_65/StatefulPartitionedCallв.batch_normalization_66/StatefulPartitionedCallв.batch_normalization_67/StatefulPartitionedCallв!conv2d_45/StatefulPartitionedCallв!conv2d_46/StatefulPartitionedCallв!conv2d_47/StatefulPartitionedCallв!conv2d_48/StatefulPartitionedCallв!conv2d_49/StatefulPartitionedCallв!conv2d_50/StatefulPartitionedCallв dense_30/StatefulPartitionedCallв dense_31/StatefulPartitionedCallв dense_32/StatefulPartitionedCallв dense_33/StatefulPartitionedCallв"dropout_60/StatefulPartitionedCallв"dropout_61/StatefulPartitionedCallв"dropout_62/StatefulPartitionedCallв"dropout_63/StatefulPartitionedCallв"dropout_64/StatefulPartitionedCallв"dropout_65/StatefulPartitionedCallв"dropout_66/StatefulPartitionedCallв"dropout_67/StatefulPartitionedCallд
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCallinput_6conv2d_48_38839conv2d_48_38841*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_48_layer_call_and_return_conditional_losses_388282#
!conv2d_48/StatefulPartitionedCallд
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinput_6conv2d_45_38865conv2d_45_38867*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_388542#
!conv2d_45/StatefulPartitionedCallУ
activation_64/PartitionedCallPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_64_layer_call_and_return_conditional_losses_388752
activation_64/PartitionedCallУ
activation_60/PartitionedCallPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_60_layer_call_and_return_conditional_losses_388882
activation_60/PartitionedCall┬
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall&activation_64/PartitionedCall:output:0batch_normalization_64_38960batch_normalization_64_38962batch_normalization_64_38964batch_normalization_64_38966*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3891520
.batch_normalization_64/StatefulPartitionedCall┬
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall&activation_60/PartitionedCall:output:0batch_normalization_60_39033batch_normalization_60_39035batch_normalization_60_39037batch_normalization_60_39039*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_3898820
.batch_normalization_60/StatefulPartitionedCallз
 max_pooling2d_48/PartitionedCallPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_380642"
 max_pooling2d_48/PartitionedCallз
 max_pooling2d_45/PartitionedCallPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_380522"
 max_pooling2d_45/PartitionedCallЯ
"dropout_64/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_64_layer_call_and_return_conditional_losses_390562$
"dropout_64/StatefulPartitionedCall─
"dropout_60/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0#^dropout_64/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_60_layer_call_and_return_conditional_losses_390862$
"dropout_60/StatefulPartitionedCall╞
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall+dropout_64/StatefulPartitionedCall:output:0conv2d_49_39125conv2d_49_39127*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_391142#
!conv2d_49/StatefulPartitionedCall╞
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall+dropout_60/StatefulPartitionedCall:output:0conv2d_46_39151conv2d_46_39153*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_46_layer_call_and_return_conditional_losses_391402#
!conv2d_46/StatefulPartitionedCallС
activation_65/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_65_layer_call_and_return_conditional_losses_391612
activation_65/PartitionedCallС
activation_61/PartitionedCallPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_61_layer_call_and_return_conditional_losses_391742
activation_61/PartitionedCall└
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall&activation_65/PartitionedCall:output:0batch_normalization_65_39246batch_normalization_65_39248batch_normalization_65_39250batch_normalization_65_39252*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3920120
.batch_normalization_65/StatefulPartitionedCall└
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall&activation_61/PartitionedCall:output:0batch_normalization_61_39319batch_normalization_61_39321batch_normalization_61_39323batch_normalization_61_39325*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_3927420
.batch_normalization_61/StatefulPartitionedCallз
 max_pooling2d_49/PartitionedCallPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_382962"
 max_pooling2d_49/PartitionedCallз
 max_pooling2d_46/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_382842"
 max_pooling2d_46/PartitionedCall─
"dropout_65/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0#^dropout_60/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_65_layer_call_and_return_conditional_losses_393422$
"dropout_65/StatefulPartitionedCall─
"dropout_61/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0#^dropout_65/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_61_layer_call_and_return_conditional_losses_393722$
"dropout_61/StatefulPartitionedCall╞
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall+dropout_65/StatefulPartitionedCall:output:0conv2d_50_39411conv2d_50_39413*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_394002#
!conv2d_50/StatefulPartitionedCall╞
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall+dropout_61/StatefulPartitionedCall:output:0conv2d_47_39437conv2d_47_39439*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_394262#
!conv2d_47/StatefulPartitionedCallС
activation_66/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_66_layer_call_and_return_conditional_losses_394472
activation_66/PartitionedCallС
activation_62/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_62_layer_call_and_return_conditional_losses_394602
activation_62/PartitionedCall└
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall&activation_66/PartitionedCall:output:0batch_normalization_66_39532batch_normalization_66_39534batch_normalization_66_39536batch_normalization_66_39538*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3948720
.batch_normalization_66/StatefulPartitionedCall└
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall&activation_62/PartitionedCall:output:0batch_normalization_62_39605batch_normalization_62_39607batch_normalization_62_39609batch_normalization_62_39611*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_3956020
.batch_normalization_62/StatefulPartitionedCallз
 max_pooling2d_50/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_385282"
 max_pooling2d_50/PartitionedCallз
 max_pooling2d_47/PartitionedCallPartitionedCall7batch_normalization_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_385162"
 max_pooling2d_47/PartitionedCall─
"dropout_66/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0#^dropout_61/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_66_layer_call_and_return_conditional_losses_396282$
"dropout_66/StatefulPartitionedCall─
"dropout_62/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0#^dropout_66/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_62_layer_call_and_return_conditional_losses_396582$
"dropout_62/StatefulPartitionedCallВ
flatten_16/PartitionedCallPartitionedCall+dropout_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_396822
flatten_16/PartitionedCallВ
flatten_15/PartitionedCallPartitionedCall+dropout_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_15_layer_call_and_return_conditional_losses_396962
flatten_15/PartitionedCall▓
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_39725dense_32_39727*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_397142"
 dense_32/StatefulPartitionedCall▓
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_30_39751dense_30_39753*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_397402"
 dense_30/StatefulPartitionedCallЙ
activation_67/PartitionedCallPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_67_layer_call_and_return_conditional_losses_397612
activation_67/PartitionedCallЙ
activation_63/PartitionedCallPartitionedCall)dense_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_63_layer_call_and_return_conditional_losses_397742
activation_63/PartitionedCall╣
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall&activation_67/PartitionedCall:output:0batch_normalization_67_39808batch_normalization_67_39810batch_normalization_67_39812batch_normalization_67_39814*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3877020
.batch_normalization_67/StatefulPartitionedCall╣
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall&activation_63/PartitionedCall:output:0batch_normalization_63_39843batch_normalization_63_39845batch_normalization_63_39847batch_normalization_63_39849*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3863020
.batch_normalization_63/StatefulPartitionedCall╦
"dropout_67/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0#^dropout_62/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_67_layer_call_and_return_conditional_losses_398642$
"dropout_67/StatefulPartitionedCall╦
"dropout_63/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0#^dropout_67/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_63_layer_call_and_return_conditional_losses_398942$
"dropout_63/StatefulPartitionedCall╣
 dense_33/StatefulPartitionedCallStatefulPartitionedCall+dropout_67/StatefulPartitionedCall:output:0dense_33_39933dense_33_39935*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_399222"
 dense_33/StatefulPartitionedCall╣
 dense_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_63/StatefulPartitionedCall:output:0dense_31_39959dense_31_39961*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_399482"
 dense_31/StatefulPartitionedCallИ
gender_output/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_gender_output_layer_call_and_return_conditional_losses_399692
gender_output/PartitionedCall 
age_output/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_age_output_layer_call_and_return_conditional_losses_399812
age_output/PartitionedCallЛ	
IdentityIdentity#age_output/PartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall#^dropout_60/StatefulPartitionedCall#^dropout_61/StatefulPartitionedCall#^dropout_62/StatefulPartitionedCall#^dropout_63/StatefulPartitionedCall#^dropout_64/StatefulPartitionedCall#^dropout_65/StatefulPartitionedCall#^dropout_66/StatefulPartitionedCall#^dropout_67/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ	

Identity_1Identity&gender_output/PartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall#^dropout_60/StatefulPartitionedCall#^dropout_61/StatefulPartitionedCall#^dropout_62/StatefulPartitionedCall#^dropout_63/StatefulPartitionedCall#^dropout_64/StatefulPartitionedCall#^dropout_65/StatefulPartitionedCall#^dropout_66/StatefulPartitionedCall#^dropout_67/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2H
"dropout_60/StatefulPartitionedCall"dropout_60/StatefulPartitionedCall2H
"dropout_61/StatefulPartitionedCall"dropout_61/StatefulPartitionedCall2H
"dropout_62/StatefulPartitionedCall"dropout_62/StatefulPartitionedCall2H
"dropout_63/StatefulPartitionedCall"dropout_63/StatefulPartitionedCall2H
"dropout_64/StatefulPartitionedCall"dropout_64/StatefulPartitionedCall2H
"dropout_65/StatefulPartitionedCall"dropout_65/StatefulPartitionedCall2H
"dropout_66/StatefulPartitionedCall"dropout_66/StatefulPartitionedCall2H
"dropout_67/StatefulPartitionedCall"dropout_67/StatefulPartitionedCall:Z V
1
_output_shapes
:         ╞╞
!
_user_specified_name	input_6
╖
F
*__inference_dropout_61_layer_call_fn_42241

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_61_layer_call_and_return_conditional_losses_393772
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
╠
c
E__inference_dropout_67_layer_call_and_return_conditional_losses_42924

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
├
d
E__inference_dropout_62_layer_call_and_return_conditional_losses_42594

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╠
c
E__inference_dropout_63_layer_call_and_return_conditional_losses_42897

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┴
Ї
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41820

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╓
d
H__inference_activation_62_layer_call_and_return_conditional_losses_39460

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         !! 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
ш
c
E__inference_dropout_60_layer_call_and_return_conditional_losses_41863

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         BB2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         BB2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         BB:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
Б
Ї
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41756

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3▄
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
т
И
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_42772

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
║
d
H__inference_activation_63_layer_call_and_return_conditional_losses_39774

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Н
Ш
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_38988

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1А
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
Б
Ї
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_39006

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3▄
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
├
d
E__inference_dropout_64_layer_call_and_return_conditional_losses_39056

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         BB2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         BB*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         BB2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         BB2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         BB2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         BB2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
Щ	
▄
C__inference_dense_32_layer_call_and_return_conditional_losses_39714

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
З
~
)__inference_conv2d_45_layer_call_fn_41551

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_388542
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╞╞::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
╓
d
H__inference_activation_65_layer_call_and_return_conditional_losses_41953

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         BB 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB :W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_38052

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
в
й
6__inference_batch_normalization_61_layer_call_fn_42022

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_381632
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
С0
╞
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_42834

inputs
assignmovingavg_42809
assignmovingavg_1_42815)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/42809*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_42809*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/42809*
_output_shapes	
:А2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/42809*
_output_shapes	
:А2
AssignMovingAvg/mulн
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_42809AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/42809*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╤
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/42815*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_42815*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp√
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/42815*
_output_shapes	
:А2
AssignMovingAvg_1/subЄ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/42815*
_output_shapes	
:А2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_42815AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/42815*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Е
Ш
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42346

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
К
d
E__inference_dropout_67_layer_call_and_return_conditional_losses_42919

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ы
F
*__inference_dropout_63_layer_call_fn_42907

inputs
identity╟
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_63_layer_call_and_return_conditional_losses_398992
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╓
d
H__inference_activation_61_layer_call_and_return_conditional_losses_41943

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         BB 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB :W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
┐╟
║
C__inference_face_net_layer_call_and_return_conditional_losses_40564

inputs
conv2d_48_40414
conv2d_48_40416
conv2d_45_40419
conv2d_45_40421 
batch_normalization_64_40426 
batch_normalization_64_40428 
batch_normalization_64_40430 
batch_normalization_64_40432 
batch_normalization_60_40435 
batch_normalization_60_40437 
batch_normalization_60_40439 
batch_normalization_60_40441
conv2d_49_40448
conv2d_49_40450
conv2d_46_40453
conv2d_46_40455 
batch_normalization_65_40460 
batch_normalization_65_40462 
batch_normalization_65_40464 
batch_normalization_65_40466 
batch_normalization_61_40469 
batch_normalization_61_40471 
batch_normalization_61_40473 
batch_normalization_61_40475
conv2d_50_40482
conv2d_50_40484
conv2d_47_40487
conv2d_47_40489 
batch_normalization_66_40494 
batch_normalization_66_40496 
batch_normalization_66_40498 
batch_normalization_66_40500 
batch_normalization_62_40503 
batch_normalization_62_40505 
batch_normalization_62_40507 
batch_normalization_62_40509
dense_32_40518
dense_32_40520
dense_30_40523
dense_30_40525 
batch_normalization_67_40530 
batch_normalization_67_40532 
batch_normalization_67_40534 
batch_normalization_67_40536 
batch_normalization_63_40539 
batch_normalization_63_40541 
batch_normalization_63_40543 
batch_normalization_63_40545
dense_33_40550
dense_33_40552
dense_31_40555
dense_31_40557
identity

identity_1Ив.batch_normalization_60/StatefulPartitionedCallв.batch_normalization_61/StatefulPartitionedCallв.batch_normalization_62/StatefulPartitionedCallв.batch_normalization_63/StatefulPartitionedCallв.batch_normalization_64/StatefulPartitionedCallв.batch_normalization_65/StatefulPartitionedCallв.batch_normalization_66/StatefulPartitionedCallв.batch_normalization_67/StatefulPartitionedCallв!conv2d_45/StatefulPartitionedCallв!conv2d_46/StatefulPartitionedCallв!conv2d_47/StatefulPartitionedCallв!conv2d_48/StatefulPartitionedCallв!conv2d_49/StatefulPartitionedCallв!conv2d_50/StatefulPartitionedCallв dense_30/StatefulPartitionedCallв dense_31/StatefulPartitionedCallв dense_32/StatefulPartitionedCallв dense_33/StatefulPartitionedCallг
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_48_40414conv2d_48_40416*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_48_layer_call_and_return_conditional_losses_388282#
!conv2d_48/StatefulPartitionedCallг
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_45_40419conv2d_45_40421*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_388542#
!conv2d_45/StatefulPartitionedCallУ
activation_64/PartitionedCallPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_64_layer_call_and_return_conditional_losses_388752
activation_64/PartitionedCallУ
activation_60/PartitionedCallPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_60_layer_call_and_return_conditional_losses_388882
activation_60/PartitionedCall─
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall&activation_64/PartitionedCall:output:0batch_normalization_64_40426batch_normalization_64_40428batch_normalization_64_40430batch_normalization_64_40432*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3893320
.batch_normalization_64/StatefulPartitionedCall─
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall&activation_60/PartitionedCall:output:0batch_normalization_60_40435batch_normalization_60_40437batch_normalization_60_40439batch_normalization_60_40441*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_3900620
.batch_normalization_60/StatefulPartitionedCallз
 max_pooling2d_48/PartitionedCallPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_380642"
 max_pooling2d_48/PartitionedCallз
 max_pooling2d_45/PartitionedCallPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_380522"
 max_pooling2d_45/PartitionedCallЗ
dropout_64/PartitionedCallPartitionedCall)max_pooling2d_48/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_64_layer_call_and_return_conditional_losses_390612
dropout_64/PartitionedCallЗ
dropout_60/PartitionedCallPartitionedCall)max_pooling2d_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_60_layer_call_and_return_conditional_losses_390912
dropout_60/PartitionedCall╛
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall#dropout_64/PartitionedCall:output:0conv2d_49_40448conv2d_49_40450*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_391142#
!conv2d_49/StatefulPartitionedCall╛
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall#dropout_60/PartitionedCall:output:0conv2d_46_40453conv2d_46_40455*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_46_layer_call_and_return_conditional_losses_391402#
!conv2d_46/StatefulPartitionedCallС
activation_65/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_65_layer_call_and_return_conditional_losses_391612
activation_65/PartitionedCallС
activation_61/PartitionedCallPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_61_layer_call_and_return_conditional_losses_391742
activation_61/PartitionedCall┬
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall&activation_65/PartitionedCall:output:0batch_normalization_65_40460batch_normalization_65_40462batch_normalization_65_40464batch_normalization_65_40466*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3921920
.batch_normalization_65/StatefulPartitionedCall┬
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall&activation_61/PartitionedCall:output:0batch_normalization_61_40469batch_normalization_61_40471batch_normalization_61_40473batch_normalization_61_40475*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_3929220
.batch_normalization_61/StatefulPartitionedCallз
 max_pooling2d_49/PartitionedCallPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_382962"
 max_pooling2d_49/PartitionedCallз
 max_pooling2d_46/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_382842"
 max_pooling2d_46/PartitionedCallЗ
dropout_65/PartitionedCallPartitionedCall)max_pooling2d_49/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_65_layer_call_and_return_conditional_losses_393472
dropout_65/PartitionedCallЗ
dropout_61/PartitionedCallPartitionedCall)max_pooling2d_46/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_61_layer_call_and_return_conditional_losses_393772
dropout_61/PartitionedCall╛
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall#dropout_65/PartitionedCall:output:0conv2d_50_40482conv2d_50_40484*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_394002#
!conv2d_50/StatefulPartitionedCall╛
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall#dropout_61/PartitionedCall:output:0conv2d_47_40487conv2d_47_40489*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_394262#
!conv2d_47/StatefulPartitionedCallС
activation_66/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_66_layer_call_and_return_conditional_losses_394472
activation_66/PartitionedCallС
activation_62/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_62_layer_call_and_return_conditional_losses_394602
activation_62/PartitionedCall┬
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall&activation_66/PartitionedCall:output:0batch_normalization_66_40494batch_normalization_66_40496batch_normalization_66_40498batch_normalization_66_40500*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3950520
.batch_normalization_66/StatefulPartitionedCall┬
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall&activation_62/PartitionedCall:output:0batch_normalization_62_40503batch_normalization_62_40505batch_normalization_62_40507batch_normalization_62_40509*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_3957820
.batch_normalization_62/StatefulPartitionedCallз
 max_pooling2d_50/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_385282"
 max_pooling2d_50/PartitionedCallз
 max_pooling2d_47/PartitionedCallPartitionedCall7batch_normalization_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_385162"
 max_pooling2d_47/PartitionedCallЗ
dropout_66/PartitionedCallPartitionedCall)max_pooling2d_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_66_layer_call_and_return_conditional_losses_396332
dropout_66/PartitionedCallЗ
dropout_62/PartitionedCallPartitionedCall)max_pooling2d_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_62_layer_call_and_return_conditional_losses_396632
dropout_62/PartitionedCall·
flatten_16/PartitionedCallPartitionedCall#dropout_66/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_396822
flatten_16/PartitionedCall·
flatten_15/PartitionedCallPartitionedCall#dropout_62/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_15_layer_call_and_return_conditional_losses_396962
flatten_15/PartitionedCall▓
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_40518dense_32_40520*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_397142"
 dense_32/StatefulPartitionedCall▓
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_30_40523dense_30_40525*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_397402"
 dense_30/StatefulPartitionedCallЙ
activation_67/PartitionedCallPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_67_layer_call_and_return_conditional_losses_397612
activation_67/PartitionedCallЙ
activation_63/PartitionedCallPartitionedCall)dense_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_63_layer_call_and_return_conditional_losses_397742
activation_63/PartitionedCall╗
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall&activation_67/PartitionedCall:output:0batch_normalization_67_40530batch_normalization_67_40532batch_normalization_67_40534batch_normalization_67_40536*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3880320
.batch_normalization_67/StatefulPartitionedCall╗
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall&activation_63/PartitionedCall:output:0batch_normalization_63_40539batch_normalization_63_40541batch_normalization_63_40543batch_normalization_63_40545*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3866320
.batch_normalization_63/StatefulPartitionedCallО
dropout_67/PartitionedCallPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_67_layer_call_and_return_conditional_losses_398692
dropout_67/PartitionedCallО
dropout_63/PartitionedCallPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_63_layer_call_and_return_conditional_losses_398992
dropout_63/PartitionedCall▒
 dense_33/StatefulPartitionedCallStatefulPartitionedCall#dropout_67/PartitionedCall:output:0dense_33_40550dense_33_40552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_399222"
 dense_33/StatefulPartitionedCall▒
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_63/PartitionedCall:output:0dense_31_40555dense_31_40557*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_399482"
 dense_31/StatefulPartitionedCallИ
gender_output/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_gender_output_layer_call_and_return_conditional_losses_399692
gender_output/PartitionedCall 
age_output/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_age_output_layer_call_and_return_conditional_losses_399812
age_output/PartitionedCallу
IdentityIdentity#age_output/PartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityъ

Identity_1Identity&gender_output/PartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
╜
a
E__inference_flatten_16_layer_call_and_return_conditional_losses_39682

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ф	
▄
C__inference_dense_33_layer_call_and_return_conditional_losses_42963

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╠
c
E__inference_dropout_63_layer_call_and_return_conditional_losses_39899

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Е
Ш
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42170

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
З
~
)__inference_conv2d_48_layer_call_fn_41570

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_48_layer_call_and_return_conditional_losses_388282
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╞╞::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
╓
е
(__inference_face_net_layer_call_fn_40409
input_6
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity

identity_1ИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *F
_read_only_resource_inputs(
&$	
!"%&'(+,/01234*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_face_net_layer_call_and_return_conditional_losses_403002
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ╞╞
!
_user_specified_name	input_6
Е
Ш
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_42042

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
║
d
H__inference_activation_67_layer_call_and_return_conditional_losses_39761

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Э
I
-__inference_gender_output_layer_call_fn_42991

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_gender_output_layer_call_and_return_conditional_losses_399692
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Е
Ш
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_39201

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
┼
I
-__inference_activation_60_layer_call_fn_41580

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_60_layer_call_and_return_conditional_losses_388882
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ╞╞:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
╜
a
E__inference_flatten_16_layer_call_and_return_conditional_losses_42653

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
├
d
E__inference_dropout_65_layer_call_and_return_conditional_losses_42253

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         !! 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         !! *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         !! 2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         !! 2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         !! 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
Б
Ї
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_38933

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3▄
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
╕
d
H__inference_gender_output_layer_call_and_return_conditional_losses_42986

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ї	
▌
D__inference_conv2d_48_layer_call_and_return_conditional_losses_41561

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpе
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞2	
BiasAddЯ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╞╞::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
╪
й
6__inference_batch_normalization_61_layer_call_fn_42073

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_392742
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
Щ	
▄
C__inference_dense_30_layer_call_and_return_conditional_losses_42668

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
∙
Ї
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_39292

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
т
й
6__inference_batch_normalization_64_layer_call_fn_41782

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_389332
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
К
d
E__inference_dropout_67_layer_call_and_return_conditional_losses_39864

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╖
F
*__inference_dropout_60_layer_call_fn_41873

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_60_layer_call_and_return_conditional_losses_390912
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         BB2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
 
~
)__inference_conv2d_47_layer_call_fn_42287

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_394262
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         !! ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
з
c
*__inference_dropout_67_layer_call_fn_42929

inputs
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_67_layer_call_and_return_conditional_losses_398642
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
▐
d
H__inference_activation_60_layer_call_and_return_conditional_losses_41575

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:         ╞╞2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ╞╞:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
ъ	
▌
D__inference_conv2d_47_layer_call_and_return_conditional_losses_39426

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! 2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         !! ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
▐
d
H__inference_activation_60_layer_call_and_return_conditional_losses_38888

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:         ╞╞2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ╞╞:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
К
d
E__inference_dropout_63_layer_call_and_return_conditional_losses_42892

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ш
c
E__inference_dropout_64_layer_call_and_return_conditional_losses_41890

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         BB2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         BB2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         BB:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
ш
c
E__inference_dropout_62_layer_call_and_return_conditional_losses_39663

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:          2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_38468

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Е
Ш
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_39487

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
у
д
(__inference_face_net_layer_call_fn_41532

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity

identity_1ИвStatefulPartitionedCall╜
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_face_net_layer_call_and_return_conditional_losses_405642
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
╖
F
*__inference_dropout_62_layer_call_fn_42609

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_62_layer_call_and_return_conditional_losses_396632
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
┴
Ї
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_37931

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
▀
}
(__inference_dense_33_layer_call_fn_42972

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_399222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
·╓
т
C__inference_face_net_layer_call_and_return_conditional_losses_40300

inputs
conv2d_48_40150
conv2d_48_40152
conv2d_45_40155
conv2d_45_40157 
batch_normalization_64_40162 
batch_normalization_64_40164 
batch_normalization_64_40166 
batch_normalization_64_40168 
batch_normalization_60_40171 
batch_normalization_60_40173 
batch_normalization_60_40175 
batch_normalization_60_40177
conv2d_49_40184
conv2d_49_40186
conv2d_46_40189
conv2d_46_40191 
batch_normalization_65_40196 
batch_normalization_65_40198 
batch_normalization_65_40200 
batch_normalization_65_40202 
batch_normalization_61_40205 
batch_normalization_61_40207 
batch_normalization_61_40209 
batch_normalization_61_40211
conv2d_50_40218
conv2d_50_40220
conv2d_47_40223
conv2d_47_40225 
batch_normalization_66_40230 
batch_normalization_66_40232 
batch_normalization_66_40234 
batch_normalization_66_40236 
batch_normalization_62_40239 
batch_normalization_62_40241 
batch_normalization_62_40243 
batch_normalization_62_40245
dense_32_40254
dense_32_40256
dense_30_40259
dense_30_40261 
batch_normalization_67_40266 
batch_normalization_67_40268 
batch_normalization_67_40270 
batch_normalization_67_40272 
batch_normalization_63_40275 
batch_normalization_63_40277 
batch_normalization_63_40279 
batch_normalization_63_40281
dense_33_40286
dense_33_40288
dense_31_40291
dense_31_40293
identity

identity_1Ив.batch_normalization_60/StatefulPartitionedCallв.batch_normalization_61/StatefulPartitionedCallв.batch_normalization_62/StatefulPartitionedCallв.batch_normalization_63/StatefulPartitionedCallв.batch_normalization_64/StatefulPartitionedCallв.batch_normalization_65/StatefulPartitionedCallв.batch_normalization_66/StatefulPartitionedCallв.batch_normalization_67/StatefulPartitionedCallв!conv2d_45/StatefulPartitionedCallв!conv2d_46/StatefulPartitionedCallв!conv2d_47/StatefulPartitionedCallв!conv2d_48/StatefulPartitionedCallв!conv2d_49/StatefulPartitionedCallв!conv2d_50/StatefulPartitionedCallв dense_30/StatefulPartitionedCallв dense_31/StatefulPartitionedCallв dense_32/StatefulPartitionedCallв dense_33/StatefulPartitionedCallв"dropout_60/StatefulPartitionedCallв"dropout_61/StatefulPartitionedCallв"dropout_62/StatefulPartitionedCallв"dropout_63/StatefulPartitionedCallв"dropout_64/StatefulPartitionedCallв"dropout_65/StatefulPartitionedCallв"dropout_66/StatefulPartitionedCallв"dropout_67/StatefulPartitionedCallг
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_48_40150conv2d_48_40152*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_48_layer_call_and_return_conditional_losses_388282#
!conv2d_48/StatefulPartitionedCallг
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_45_40155conv2d_45_40157*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_388542#
!conv2d_45/StatefulPartitionedCallУ
activation_64/PartitionedCallPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_64_layer_call_and_return_conditional_losses_388752
activation_64/PartitionedCallУ
activation_60/PartitionedCallPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_60_layer_call_and_return_conditional_losses_388882
activation_60/PartitionedCall┬
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall&activation_64/PartitionedCall:output:0batch_normalization_64_40162batch_normalization_64_40164batch_normalization_64_40166batch_normalization_64_40168*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3891520
.batch_normalization_64/StatefulPartitionedCall┬
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall&activation_60/PartitionedCall:output:0batch_normalization_60_40171batch_normalization_60_40173batch_normalization_60_40175batch_normalization_60_40177*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_3898820
.batch_normalization_60/StatefulPartitionedCallз
 max_pooling2d_48/PartitionedCallPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_380642"
 max_pooling2d_48/PartitionedCallз
 max_pooling2d_45/PartitionedCallPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_380522"
 max_pooling2d_45/PartitionedCallЯ
"dropout_64/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_48/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_64_layer_call_and_return_conditional_losses_390562$
"dropout_64/StatefulPartitionedCall─
"dropout_60/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_45/PartitionedCall:output:0#^dropout_64/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_60_layer_call_and_return_conditional_losses_390862$
"dropout_60/StatefulPartitionedCall╞
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall+dropout_64/StatefulPartitionedCall:output:0conv2d_49_40184conv2d_49_40186*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_391142#
!conv2d_49/StatefulPartitionedCall╞
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall+dropout_60/StatefulPartitionedCall:output:0conv2d_46_40189conv2d_46_40191*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_46_layer_call_and_return_conditional_losses_391402#
!conv2d_46/StatefulPartitionedCallС
activation_65/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_65_layer_call_and_return_conditional_losses_391612
activation_65/PartitionedCallС
activation_61/PartitionedCallPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_61_layer_call_and_return_conditional_losses_391742
activation_61/PartitionedCall└
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall&activation_65/PartitionedCall:output:0batch_normalization_65_40196batch_normalization_65_40198batch_normalization_65_40200batch_normalization_65_40202*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3920120
.batch_normalization_65/StatefulPartitionedCall└
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall&activation_61/PartitionedCall:output:0batch_normalization_61_40205batch_normalization_61_40207batch_normalization_61_40209batch_normalization_61_40211*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_3927420
.batch_normalization_61/StatefulPartitionedCallз
 max_pooling2d_49/PartitionedCallPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_382962"
 max_pooling2d_49/PartitionedCallз
 max_pooling2d_46/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_382842"
 max_pooling2d_46/PartitionedCall─
"dropout_65/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_49/PartitionedCall:output:0#^dropout_60/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_65_layer_call_and_return_conditional_losses_393422$
"dropout_65/StatefulPartitionedCall─
"dropout_61/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_46/PartitionedCall:output:0#^dropout_65/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_61_layer_call_and_return_conditional_losses_393722$
"dropout_61/StatefulPartitionedCall╞
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall+dropout_65/StatefulPartitionedCall:output:0conv2d_50_40218conv2d_50_40220*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_394002#
!conv2d_50/StatefulPartitionedCall╞
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall+dropout_61/StatefulPartitionedCall:output:0conv2d_47_40223conv2d_47_40225*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_394262#
!conv2d_47/StatefulPartitionedCallС
activation_66/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_66_layer_call_and_return_conditional_losses_394472
activation_66/PartitionedCallС
activation_62/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_62_layer_call_and_return_conditional_losses_394602
activation_62/PartitionedCall└
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall&activation_66/PartitionedCall:output:0batch_normalization_66_40230batch_normalization_66_40232batch_normalization_66_40234batch_normalization_66_40236*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3948720
.batch_normalization_66/StatefulPartitionedCall└
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall&activation_62/PartitionedCall:output:0batch_normalization_62_40239batch_normalization_62_40241batch_normalization_62_40243batch_normalization_62_40245*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_3956020
.batch_normalization_62/StatefulPartitionedCallз
 max_pooling2d_50/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_385282"
 max_pooling2d_50/PartitionedCallз
 max_pooling2d_47/PartitionedCallPartitionedCall7batch_normalization_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_385162"
 max_pooling2d_47/PartitionedCall─
"dropout_66/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_50/PartitionedCall:output:0#^dropout_61/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_66_layer_call_and_return_conditional_losses_396282$
"dropout_66/StatefulPartitionedCall─
"dropout_62/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_47/PartitionedCall:output:0#^dropout_66/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_62_layer_call_and_return_conditional_losses_396582$
"dropout_62/StatefulPartitionedCallВ
flatten_16/PartitionedCallPartitionedCall+dropout_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_396822
flatten_16/PartitionedCallВ
flatten_15/PartitionedCallPartitionedCall+dropout_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_15_layer_call_and_return_conditional_losses_396962
flatten_15/PartitionedCall▓
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_40254dense_32_40256*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_397142"
 dense_32/StatefulPartitionedCall▓
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_30_40259dense_30_40261*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_397402"
 dense_30/StatefulPartitionedCallЙ
activation_67/PartitionedCallPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_67_layer_call_and_return_conditional_losses_397612
activation_67/PartitionedCallЙ
activation_63/PartitionedCallPartitionedCall)dense_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_63_layer_call_and_return_conditional_losses_397742
activation_63/PartitionedCall╣
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall&activation_67/PartitionedCall:output:0batch_normalization_67_40266batch_normalization_67_40268batch_normalization_67_40270batch_normalization_67_40272*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3877020
.batch_normalization_67/StatefulPartitionedCall╣
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall&activation_63/PartitionedCall:output:0batch_normalization_63_40275batch_normalization_63_40277batch_normalization_63_40279batch_normalization_63_40281*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3863020
.batch_normalization_63/StatefulPartitionedCall╦
"dropout_67/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0#^dropout_62/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_67_layer_call_and_return_conditional_losses_398642$
"dropout_67/StatefulPartitionedCall╦
"dropout_63/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0#^dropout_67/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_63_layer_call_and_return_conditional_losses_398942$
"dropout_63/StatefulPartitionedCall╣
 dense_33/StatefulPartitionedCallStatefulPartitionedCall+dropout_67/StatefulPartitionedCall:output:0dense_33_40286dense_33_40288*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_399222"
 dense_33/StatefulPartitionedCall╣
 dense_31/StatefulPartitionedCallStatefulPartitionedCall+dropout_63/StatefulPartitionedCall:output:0dense_31_40291dense_31_40293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_399482"
 dense_31/StatefulPartitionedCallИ
gender_output/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_gender_output_layer_call_and_return_conditional_losses_399692
gender_output/PartitionedCall 
age_output/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_age_output_layer_call_and_return_conditional_losses_399812
age_output/PartitionedCallЛ	
IdentityIdentity#age_output/PartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall#^dropout_60/StatefulPartitionedCall#^dropout_61/StatefulPartitionedCall#^dropout_62/StatefulPartitionedCall#^dropout_63/StatefulPartitionedCall#^dropout_64/StatefulPartitionedCall#^dropout_65/StatefulPartitionedCall#^dropout_66/StatefulPartitionedCall#^dropout_67/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ	

Identity_1Identity&gender_output/PartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall#^dropout_60/StatefulPartitionedCall#^dropout_61/StatefulPartitionedCall#^dropout_62/StatefulPartitionedCall#^dropout_63/StatefulPartitionedCall#^dropout_64/StatefulPartitionedCall#^dropout_65/StatefulPartitionedCall#^dropout_66/StatefulPartitionedCall#^dropout_67/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2H
"dropout_60/StatefulPartitionedCall"dropout_60/StatefulPartitionedCall2H
"dropout_61/StatefulPartitionedCall"dropout_61/StatefulPartitionedCall2H
"dropout_62/StatefulPartitionedCall"dropout_62/StatefulPartitionedCall2H
"dropout_63/StatefulPartitionedCall"dropout_63/StatefulPartitionedCall2H
"dropout_64/StatefulPartitionedCall"dropout_64/StatefulPartitionedCall2H
"dropout_65/StatefulPartitionedCall"dropout_65/StatefulPartitionedCall2H
"dropout_66/StatefulPartitionedCall"dropout_66/StatefulPartitionedCall2H
"dropout_67/StatefulPartitionedCall"dropout_67/StatefulPartitionedCall:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_38528

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ъ	
▌
D__inference_conv2d_50_layer_call_and_return_conditional_losses_42297

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! 2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         !! ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
ъ	
▌
D__inference_conv2d_46_layer_call_and_return_conditional_losses_41910

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB 2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         BB::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
Т╕
╛)
C__inference_face_net_layer_call_and_return_conditional_losses_41310

inputs,
(conv2d_48_conv2d_readvariableop_resource-
)conv2d_48_biasadd_readvariableop_resource,
(conv2d_45_conv2d_readvariableop_resource-
)conv2d_45_biasadd_readvariableop_resource2
.batch_normalization_64_readvariableop_resource4
0batch_normalization_64_readvariableop_1_resourceC
?batch_normalization_64_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_64_fusedbatchnormv3_readvariableop_1_resource2
.batch_normalization_60_readvariableop_resource4
0batch_normalization_60_readvariableop_1_resourceC
?batch_normalization_60_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_60_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_49_conv2d_readvariableop_resource-
)conv2d_49_biasadd_readvariableop_resource,
(conv2d_46_conv2d_readvariableop_resource-
)conv2d_46_biasadd_readvariableop_resource2
.batch_normalization_65_readvariableop_resource4
0batch_normalization_65_readvariableop_1_resourceC
?batch_normalization_65_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_65_fusedbatchnormv3_readvariableop_1_resource2
.batch_normalization_61_readvariableop_resource4
0batch_normalization_61_readvariableop_1_resourceC
?batch_normalization_61_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_61_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_50_conv2d_readvariableop_resource-
)conv2d_50_biasadd_readvariableop_resource,
(conv2d_47_conv2d_readvariableop_resource-
)conv2d_47_biasadd_readvariableop_resource2
.batch_normalization_66_readvariableop_resource4
0batch_normalization_66_readvariableop_1_resourceC
?batch_normalization_66_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource2
.batch_normalization_62_readvariableop_resource4
0batch_normalization_62_readvariableop_1_resourceC
?batch_normalization_62_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_62_fusedbatchnormv3_readvariableop_1_resource+
'dense_32_matmul_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource+
'dense_30_matmul_readvariableop_resource,
(dense_30_biasadd_readvariableop_resource<
8batch_normalization_67_batchnorm_readvariableop_resource@
<batch_normalization_67_batchnorm_mul_readvariableop_resource>
:batch_normalization_67_batchnorm_readvariableop_1_resource>
:batch_normalization_67_batchnorm_readvariableop_2_resource<
8batch_normalization_63_batchnorm_readvariableop_resource@
<batch_normalization_63_batchnorm_mul_readvariableop_resource>
:batch_normalization_63_batchnorm_readvariableop_1_resource>
:batch_normalization_63_batchnorm_readvariableop_2_resource+
'dense_33_matmul_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource+
'dense_31_matmul_readvariableop_resource,
(dense_31_biasadd_readvariableop_resource
identity

identity_1Ив6batch_normalization_60/FusedBatchNormV3/ReadVariableOpв8batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_60/ReadVariableOpв'batch_normalization_60/ReadVariableOp_1в6batch_normalization_61/FusedBatchNormV3/ReadVariableOpв8batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_61/ReadVariableOpв'batch_normalization_61/ReadVariableOp_1в6batch_normalization_62/FusedBatchNormV3/ReadVariableOpв8batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_62/ReadVariableOpв'batch_normalization_62/ReadVariableOp_1в/batch_normalization_63/batchnorm/ReadVariableOpв1batch_normalization_63/batchnorm/ReadVariableOp_1в1batch_normalization_63/batchnorm/ReadVariableOp_2в3batch_normalization_63/batchnorm/mul/ReadVariableOpв6batch_normalization_64/FusedBatchNormV3/ReadVariableOpв8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_64/ReadVariableOpв'batch_normalization_64/ReadVariableOp_1в6batch_normalization_65/FusedBatchNormV3/ReadVariableOpв8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_65/ReadVariableOpв'batch_normalization_65/ReadVariableOp_1в6batch_normalization_66/FusedBatchNormV3/ReadVariableOpв8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1в%batch_normalization_66/ReadVariableOpв'batch_normalization_66/ReadVariableOp_1в/batch_normalization_67/batchnorm/ReadVariableOpв1batch_normalization_67/batchnorm/ReadVariableOp_1в1batch_normalization_67/batchnorm/ReadVariableOp_2в3batch_normalization_67/batchnorm/mul/ReadVariableOpв conv2d_45/BiasAdd/ReadVariableOpвconv2d_45/Conv2D/ReadVariableOpв conv2d_46/BiasAdd/ReadVariableOpвconv2d_46/Conv2D/ReadVariableOpв conv2d_47/BiasAdd/ReadVariableOpвconv2d_47/Conv2D/ReadVariableOpв conv2d_48/BiasAdd/ReadVariableOpвconv2d_48/Conv2D/ReadVariableOpв conv2d_49/BiasAdd/ReadVariableOpвconv2d_49/Conv2D/ReadVariableOpв conv2d_50/BiasAdd/ReadVariableOpвconv2d_50/Conv2D/ReadVariableOpвdense_30/BiasAdd/ReadVariableOpвdense_30/MatMul/ReadVariableOpвdense_31/BiasAdd/ReadVariableOpвdense_31/MatMul/ReadVariableOpвdense_32/BiasAdd/ReadVariableOpвdense_32/MatMul/ReadVariableOpвdense_33/BiasAdd/ReadVariableOpвdense_33/MatMul/ReadVariableOp│
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_48/Conv2D/ReadVariableOp├
conv2d_48/Conv2DConv2Dinputs'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞*
paddingSAME*
strides
2
conv2d_48/Conv2Dк
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp▓
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞2
conv2d_48/BiasAdd│
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_45/Conv2D/ReadVariableOp├
conv2d_45/Conv2DConv2Dinputs'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞*
paddingSAME*
strides
2
conv2d_45/Conv2Dк
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_45/BiasAdd/ReadVariableOp▓
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞2
conv2d_45/BiasAddИ
activation_64/ReluReluconv2d_48/BiasAdd:output:0*
T0*1
_output_shapes
:         ╞╞2
activation_64/ReluИ
activation_60/ReluReluconv2d_45/BiasAdd:output:0*
T0*1
_output_shapes
:         ╞╞2
activation_60/Relu╣
%batch_normalization_64/ReadVariableOpReadVariableOp.batch_normalization_64_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_64/ReadVariableOp┐
'batch_normalization_64/ReadVariableOp_1ReadVariableOp0batch_normalization_64_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_64/ReadVariableOp_1ь
6batch_normalization_64/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_64_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_64/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_64_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1Ё
'batch_normalization_64/FusedBatchNormV3FusedBatchNormV3 activation_64/Relu:activations:0-batch_normalization_64/ReadVariableOp:value:0/batch_normalization_64/ReadVariableOp_1:value:0>batch_normalization_64/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
is_training( 2)
'batch_normalization_64/FusedBatchNormV3╣
%batch_normalization_60/ReadVariableOpReadVariableOp.batch_normalization_60_readvariableop_resource*
_output_shapes
:*
dtype02'
%batch_normalization_60/ReadVariableOp┐
'batch_normalization_60/ReadVariableOp_1ReadVariableOp0batch_normalization_60_readvariableop_1_resource*
_output_shapes
:*
dtype02)
'batch_normalization_60/ReadVariableOp_1ь
6batch_normalization_60/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_60_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype028
6batch_normalization_60/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_60_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02:
8batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1Ё
'batch_normalization_60/FusedBatchNormV3FusedBatchNormV3 activation_60/Relu:activations:0-batch_normalization_60/ReadVariableOp:value:0/batch_normalization_60/ReadVariableOp_1:value:0>batch_normalization_60/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
is_training( 2)
'batch_normalization_60/FusedBatchNormV3┘
max_pooling2d_48/MaxPoolMaxPool+batch_normalization_64/FusedBatchNormV3:y:0*/
_output_shapes
:         BB*
ksize
*
paddingVALID*
strides
2
max_pooling2d_48/MaxPool┘
max_pooling2d_45/MaxPoolMaxPool+batch_normalization_60/FusedBatchNormV3:y:0*/
_output_shapes
:         BB*
ksize
*
paddingVALID*
strides
2
max_pooling2d_45/MaxPoolУ
dropout_64/IdentityIdentity!max_pooling2d_48/MaxPool:output:0*
T0*/
_output_shapes
:         BB2
dropout_64/IdentityУ
dropout_60/IdentityIdentity!max_pooling2d_45/MaxPool:output:0*
T0*/
_output_shapes
:         BB2
dropout_60/Identity│
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_49/Conv2D/ReadVariableOp╫
conv2d_49/Conv2DConv2Ddropout_64/Identity:output:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB *
paddingSAME*
strides
2
conv2d_49/Conv2Dк
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp░
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB 2
conv2d_49/BiasAdd│
conv2d_46/Conv2D/ReadVariableOpReadVariableOp(conv2d_46_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_46/Conv2D/ReadVariableOp╫
conv2d_46/Conv2DConv2Ddropout_60/Identity:output:0'conv2d_46/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB *
paddingSAME*
strides
2
conv2d_46/Conv2Dк
 conv2d_46/BiasAdd/ReadVariableOpReadVariableOp)conv2d_46_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_46/BiasAdd/ReadVariableOp░
conv2d_46/BiasAddBiasAddconv2d_46/Conv2D:output:0(conv2d_46/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB 2
conv2d_46/BiasAddЖ
activation_65/ReluReluconv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:         BB 2
activation_65/ReluЖ
activation_61/ReluReluconv2d_46/BiasAdd:output:0*
T0*/
_output_shapes
:         BB 2
activation_61/Relu╣
%batch_normalization_65/ReadVariableOpReadVariableOp.batch_normalization_65_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_65/ReadVariableOp┐
'batch_normalization_65/ReadVariableOp_1ReadVariableOp0batch_normalization_65_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_65/ReadVariableOp_1ь
6batch_normalization_65/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_65_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_65/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_65_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1ю
'batch_normalization_65/FusedBatchNormV3FusedBatchNormV3 activation_65/Relu:activations:0-batch_normalization_65/ReadVariableOp:value:0/batch_normalization_65/ReadVariableOp_1:value:0>batch_normalization_65/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
is_training( 2)
'batch_normalization_65/FusedBatchNormV3╣
%batch_normalization_61/ReadVariableOpReadVariableOp.batch_normalization_61_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_61/ReadVariableOp┐
'batch_normalization_61/ReadVariableOp_1ReadVariableOp0batch_normalization_61_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_61/ReadVariableOp_1ь
6batch_normalization_61/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_61_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_61/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_61_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1ю
'batch_normalization_61/FusedBatchNormV3FusedBatchNormV3 activation_61/Relu:activations:0-batch_normalization_61/ReadVariableOp:value:0/batch_normalization_61/ReadVariableOp_1:value:0>batch_normalization_61/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
is_training( 2)
'batch_normalization_61/FusedBatchNormV3┘
max_pooling2d_49/MaxPoolMaxPool+batch_normalization_65/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2
max_pooling2d_49/MaxPool┘
max_pooling2d_46/MaxPoolMaxPool+batch_normalization_61/FusedBatchNormV3:y:0*/
_output_shapes
:         !! *
ksize
*
paddingVALID*
strides
2
max_pooling2d_46/MaxPoolУ
dropout_65/IdentityIdentity!max_pooling2d_49/MaxPool:output:0*
T0*/
_output_shapes
:         !! 2
dropout_65/IdentityУ
dropout_61/IdentityIdentity!max_pooling2d_46/MaxPool:output:0*
T0*/
_output_shapes
:         !! 2
dropout_61/Identity│
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_50/Conv2D/ReadVariableOp╫
conv2d_50/Conv2DConv2Ddropout_65/Identity:output:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! *
paddingSAME*
strides
2
conv2d_50/Conv2Dк
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp░
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! 2
conv2d_50/BiasAdd│
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_47/Conv2D/ReadVariableOp╫
conv2d_47/Conv2DConv2Ddropout_61/Identity:output:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! *
paddingSAME*
strides
2
conv2d_47/Conv2Dк
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_47/BiasAdd/ReadVariableOp░
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! 2
conv2d_47/BiasAddЖ
activation_66/ReluReluconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:         !! 2
activation_66/ReluЖ
activation_62/ReluReluconv2d_47/BiasAdd:output:0*
T0*/
_output_shapes
:         !! 2
activation_62/Relu╣
%batch_normalization_66/ReadVariableOpReadVariableOp.batch_normalization_66_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_66/ReadVariableOp┐
'batch_normalization_66/ReadVariableOp_1ReadVariableOp0batch_normalization_66_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_66/ReadVariableOp_1ь
6batch_normalization_66/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_66_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_66/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_66_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1ю
'batch_normalization_66/FusedBatchNormV3FusedBatchNormV3 activation_66/Relu:activations:0-batch_normalization_66/ReadVariableOp:value:0/batch_normalization_66/ReadVariableOp_1:value:0>batch_normalization_66/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 2)
'batch_normalization_66/FusedBatchNormV3╣
%batch_normalization_62/ReadVariableOpReadVariableOp.batch_normalization_62_readvariableop_resource*
_output_shapes
: *
dtype02'
%batch_normalization_62/ReadVariableOp┐
'batch_normalization_62/ReadVariableOp_1ReadVariableOp0batch_normalization_62_readvariableop_1_resource*
_output_shapes
: *
dtype02)
'batch_normalization_62/ReadVariableOp_1ь
6batch_normalization_62/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_62_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype028
6batch_normalization_62/FusedBatchNormV3/ReadVariableOpЄ
8batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_62_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1ю
'batch_normalization_62/FusedBatchNormV3FusedBatchNormV3 activation_62/Relu:activations:0-batch_normalization_62/ReadVariableOp:value:0/batch_normalization_62/ReadVariableOp_1:value:0>batch_normalization_62/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 2)
'batch_normalization_62/FusedBatchNormV3┘
max_pooling2d_50/MaxPoolMaxPool+batch_normalization_66/FusedBatchNormV3:y:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_50/MaxPool┘
max_pooling2d_47/MaxPoolMaxPool+batch_normalization_62/FusedBatchNormV3:y:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
max_pooling2d_47/MaxPoolУ
dropout_66/IdentityIdentity!max_pooling2d_50/MaxPool:output:0*
T0*/
_output_shapes
:          2
dropout_66/IdentityУ
dropout_62/IdentityIdentity!max_pooling2d_47/MaxPool:output:0*
T0*/
_output_shapes
:          2
dropout_62/Identityu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_16/ConstЯ
flatten_16/ReshapeReshapedropout_66/Identity:output:0flatten_16/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_16/Reshapeu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
flatten_15/ConstЯ
flatten_15/ReshapeReshapedropout_62/Identity:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:         А@2
flatten_15/Reshapeк
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_32/MatMul/ReadVariableOpд
dense_32/MatMulMatMulflatten_16/Reshape:output:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_32/MatMulи
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_32/BiasAdd/ReadVariableOpж
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_32/BiasAddк
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02 
dense_30/MatMul/ReadVariableOpд
dense_30/MatMulMatMulflatten_15/Reshape:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_30/MatMulи
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
dense_30/BiasAdd/ReadVariableOpж
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
dense_30/BiasAdd~
activation_67/ReluReludense_32/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
activation_67/Relu~
activation_63/ReluReludense_30/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
activation_63/Relu╪
/batch_normalization_67/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_67_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/batch_normalization_67/batchnorm/ReadVariableOpХ
&batch_normalization_67/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2(
&batch_normalization_67/batchnorm/add/yх
$batch_normalization_67/batchnorm/addAddV27batch_normalization_67/batchnorm/ReadVariableOp:value:0/batch_normalization_67/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2&
$batch_normalization_67/batchnorm/addй
&batch_normalization_67/batchnorm/RsqrtRsqrt(batch_normalization_67/batchnorm/add:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_67/batchnorm/Rsqrtф
3batch_normalization_67/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_67_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3batch_normalization_67/batchnorm/mul/ReadVariableOpт
$batch_normalization_67/batchnorm/mulMul*batch_normalization_67/batchnorm/Rsqrt:y:0;batch_normalization_67/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2&
$batch_normalization_67/batchnorm/mul╓
&batch_normalization_67/batchnorm/mul_1Mul activation_67/Relu:activations:0(batch_normalization_67/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_67/batchnorm/mul_1▐
1batch_normalization_67/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_67_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype023
1batch_normalization_67/batchnorm/ReadVariableOp_1т
&batch_normalization_67/batchnorm/mul_2Mul9batch_normalization_67/batchnorm/ReadVariableOp_1:value:0(batch_normalization_67/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_67/batchnorm/mul_2▐
1batch_normalization_67/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_67_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype023
1batch_normalization_67/batchnorm/ReadVariableOp_2р
$batch_normalization_67/batchnorm/subSub9batch_normalization_67/batchnorm/ReadVariableOp_2:value:0*batch_normalization_67/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2&
$batch_normalization_67/batchnorm/subт
&batch_normalization_67/batchnorm/add_1AddV2*batch_normalization_67/batchnorm/mul_1:z:0(batch_normalization_67/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_67/batchnorm/add_1╪
/batch_normalization_63/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_63_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype021
/batch_normalization_63/batchnorm/ReadVariableOpХ
&batch_normalization_63/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2(
&batch_normalization_63/batchnorm/add/yх
$batch_normalization_63/batchnorm/addAddV27batch_normalization_63/batchnorm/ReadVariableOp:value:0/batch_normalization_63/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2&
$batch_normalization_63/batchnorm/addй
&batch_normalization_63/batchnorm/RsqrtRsqrt(batch_normalization_63/batchnorm/add:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_63/batchnorm/Rsqrtф
3batch_normalization_63/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_63_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype025
3batch_normalization_63/batchnorm/mul/ReadVariableOpт
$batch_normalization_63/batchnorm/mulMul*batch_normalization_63/batchnorm/Rsqrt:y:0;batch_normalization_63/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2&
$batch_normalization_63/batchnorm/mul╓
&batch_normalization_63/batchnorm/mul_1Mul activation_63/Relu:activations:0(batch_normalization_63/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_63/batchnorm/mul_1▐
1batch_normalization_63/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_63_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype023
1batch_normalization_63/batchnorm/ReadVariableOp_1т
&batch_normalization_63/batchnorm/mul_2Mul9batch_normalization_63/batchnorm/ReadVariableOp_1:value:0(batch_normalization_63/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2(
&batch_normalization_63/batchnorm/mul_2▐
1batch_normalization_63/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_63_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype023
1batch_normalization_63/batchnorm/ReadVariableOp_2р
$batch_normalization_63/batchnorm/subSub9batch_normalization_63/batchnorm/ReadVariableOp_2:value:0*batch_normalization_63/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2&
$batch_normalization_63/batchnorm/subт
&batch_normalization_63/batchnorm/add_1AddV2*batch_normalization_63/batchnorm/mul_1:z:0(batch_normalization_63/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2(
&batch_normalization_63/batchnorm/add_1Х
dropout_67/IdentityIdentity*batch_normalization_67/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         А2
dropout_67/IdentityХ
dropout_63/IdentityIdentity*batch_normalization_63/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         А2
dropout_63/Identityй
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_33/MatMul/ReadVariableOpд
dense_33/MatMulMatMuldropout_67/Identity:output:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_33/MatMulз
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_33/BiasAdd/ReadVariableOpе
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_33/BiasAddй
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense_31/MatMul/ReadVariableOpд
dense_31/MatMulMatMuldropout_63/Identity:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_31/MatMulз
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_31/BiasAdd/ReadVariableOpе
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_31/BiasAddЖ
gender_output/SigmoidSigmoiddense_33/BiasAdd:output:0*
T0*'
_output_shapes
:         2
gender_output/Sigmoid█
IdentityIdentitydense_31/BiasAdd:output:07^batch_normalization_60/FusedBatchNormV3/ReadVariableOp9^batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_60/ReadVariableOp(^batch_normalization_60/ReadVariableOp_17^batch_normalization_61/FusedBatchNormV3/ReadVariableOp9^batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_61/ReadVariableOp(^batch_normalization_61/ReadVariableOp_17^batch_normalization_62/FusedBatchNormV3/ReadVariableOp9^batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_62/ReadVariableOp(^batch_normalization_62/ReadVariableOp_10^batch_normalization_63/batchnorm/ReadVariableOp2^batch_normalization_63/batchnorm/ReadVariableOp_12^batch_normalization_63/batchnorm/ReadVariableOp_24^batch_normalization_63/batchnorm/mul/ReadVariableOp7^batch_normalization_64/FusedBatchNormV3/ReadVariableOp9^batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_64/ReadVariableOp(^batch_normalization_64/ReadVariableOp_17^batch_normalization_65/FusedBatchNormV3/ReadVariableOp9^batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_65/ReadVariableOp(^batch_normalization_65/ReadVariableOp_17^batch_normalization_66/FusedBatchNormV3/ReadVariableOp9^batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_66/ReadVariableOp(^batch_normalization_66/ReadVariableOp_10^batch_normalization_67/batchnorm/ReadVariableOp2^batch_normalization_67/batchnorm/ReadVariableOp_12^batch_normalization_67/batchnorm/ReadVariableOp_24^batch_normalization_67/batchnorm/mul/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity▀

Identity_1Identitygender_output/Sigmoid:y:07^batch_normalization_60/FusedBatchNormV3/ReadVariableOp9^batch_normalization_60/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_60/ReadVariableOp(^batch_normalization_60/ReadVariableOp_17^batch_normalization_61/FusedBatchNormV3/ReadVariableOp9^batch_normalization_61/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_61/ReadVariableOp(^batch_normalization_61/ReadVariableOp_17^batch_normalization_62/FusedBatchNormV3/ReadVariableOp9^batch_normalization_62/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_62/ReadVariableOp(^batch_normalization_62/ReadVariableOp_10^batch_normalization_63/batchnorm/ReadVariableOp2^batch_normalization_63/batchnorm/ReadVariableOp_12^batch_normalization_63/batchnorm/ReadVariableOp_24^batch_normalization_63/batchnorm/mul/ReadVariableOp7^batch_normalization_64/FusedBatchNormV3/ReadVariableOp9^batch_normalization_64/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_64/ReadVariableOp(^batch_normalization_64/ReadVariableOp_17^batch_normalization_65/FusedBatchNormV3/ReadVariableOp9^batch_normalization_65/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_65/ReadVariableOp(^batch_normalization_65/ReadVariableOp_17^batch_normalization_66/FusedBatchNormV3/ReadVariableOp9^batch_normalization_66/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_66/ReadVariableOp(^batch_normalization_66/ReadVariableOp_10^batch_normalization_67/batchnorm/ReadVariableOp2^batch_normalization_67/batchnorm/ReadVariableOp_12^batch_normalization_67/batchnorm/ReadVariableOp_24^batch_normalization_67/batchnorm/mul/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_46/BiasAdd/ReadVariableOp ^conv2d_46/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::2p
6batch_normalization_60/FusedBatchNormV3/ReadVariableOp6batch_normalization_60/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_60/FusedBatchNormV3/ReadVariableOp_18batch_normalization_60/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_60/ReadVariableOp%batch_normalization_60/ReadVariableOp2R
'batch_normalization_60/ReadVariableOp_1'batch_normalization_60/ReadVariableOp_12p
6batch_normalization_61/FusedBatchNormV3/ReadVariableOp6batch_normalization_61/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_61/FusedBatchNormV3/ReadVariableOp_18batch_normalization_61/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_61/ReadVariableOp%batch_normalization_61/ReadVariableOp2R
'batch_normalization_61/ReadVariableOp_1'batch_normalization_61/ReadVariableOp_12p
6batch_normalization_62/FusedBatchNormV3/ReadVariableOp6batch_normalization_62/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_62/FusedBatchNormV3/ReadVariableOp_18batch_normalization_62/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_62/ReadVariableOp%batch_normalization_62/ReadVariableOp2R
'batch_normalization_62/ReadVariableOp_1'batch_normalization_62/ReadVariableOp_12b
/batch_normalization_63/batchnorm/ReadVariableOp/batch_normalization_63/batchnorm/ReadVariableOp2f
1batch_normalization_63/batchnorm/ReadVariableOp_11batch_normalization_63/batchnorm/ReadVariableOp_12f
1batch_normalization_63/batchnorm/ReadVariableOp_21batch_normalization_63/batchnorm/ReadVariableOp_22j
3batch_normalization_63/batchnorm/mul/ReadVariableOp3batch_normalization_63/batchnorm/mul/ReadVariableOp2p
6batch_normalization_64/FusedBatchNormV3/ReadVariableOp6batch_normalization_64/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_64/FusedBatchNormV3/ReadVariableOp_18batch_normalization_64/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_64/ReadVariableOp%batch_normalization_64/ReadVariableOp2R
'batch_normalization_64/ReadVariableOp_1'batch_normalization_64/ReadVariableOp_12p
6batch_normalization_65/FusedBatchNormV3/ReadVariableOp6batch_normalization_65/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_65/FusedBatchNormV3/ReadVariableOp_18batch_normalization_65/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_65/ReadVariableOp%batch_normalization_65/ReadVariableOp2R
'batch_normalization_65/ReadVariableOp_1'batch_normalization_65/ReadVariableOp_12p
6batch_normalization_66/FusedBatchNormV3/ReadVariableOp6batch_normalization_66/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_66/FusedBatchNormV3/ReadVariableOp_18batch_normalization_66/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_66/ReadVariableOp%batch_normalization_66/ReadVariableOp2R
'batch_normalization_66/ReadVariableOp_1'batch_normalization_66/ReadVariableOp_12b
/batch_normalization_67/batchnorm/ReadVariableOp/batch_normalization_67/batchnorm/ReadVariableOp2f
1batch_normalization_67/batchnorm/ReadVariableOp_11batch_normalization_67/batchnorm/ReadVariableOp_12f
1batch_normalization_67/batchnorm/ReadVariableOp_21batch_normalization_67/batchnorm/ReadVariableOp_22j
3batch_normalization_67/batchnorm/mul/ReadVariableOp3batch_normalization_67/batchnorm/mul/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_46/BiasAdd/ReadVariableOp conv2d_46/BiasAdd/ReadVariableOp2B
conv2d_46/Conv2D/ReadVariableOpconv2d_46/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_38236

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
├
d
E__inference_dropout_65_layer_call_and_return_conditional_losses_39342

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         !! 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         !! *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         !! 2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         !! 2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         !! 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
Е
Ш
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_39274

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
ш
c
E__inference_dropout_64_layer_call_and_return_conditional_losses_39061

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         BB2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         BB2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         BB:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
╕
d
H__inference_gender_output_layer_call_and_return_conditional_losses_39969

inputs
identityW
SigmoidSigmoidinputs*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_38284

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
т
И
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_42854

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_38132

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╜
I
-__inference_activation_62_layer_call_fn_42316

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_62_layer_call_and_return_conditional_losses_394602
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
ъ	
▌
D__inference_conv2d_46_layer_call_and_return_conditional_losses_39140

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB 2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         BB::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
┌
й
6__inference_batch_normalization_62_layer_call_fn_42390

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_395782
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
┴
Ї
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_38163

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
░
L
0__inference_max_pooling2d_46_layer_call_fn_38290

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_382842
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┌
й
6__inference_batch_normalization_65_layer_call_fn_42214

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_392192
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
ы╤
╪O
!__inference__traced_restore_43859
file_prefix%
!assignvariableop_conv2d_45_kernel%
!assignvariableop_1_conv2d_45_bias'
#assignvariableop_2_conv2d_48_kernel%
!assignvariableop_3_conv2d_48_bias3
/assignvariableop_4_batch_normalization_60_gamma2
.assignvariableop_5_batch_normalization_60_beta9
5assignvariableop_6_batch_normalization_60_moving_mean=
9assignvariableop_7_batch_normalization_60_moving_variance3
/assignvariableop_8_batch_normalization_64_gamma2
.assignvariableop_9_batch_normalization_64_beta:
6assignvariableop_10_batch_normalization_64_moving_mean>
:assignvariableop_11_batch_normalization_64_moving_variance(
$assignvariableop_12_conv2d_46_kernel&
"assignvariableop_13_conv2d_46_bias(
$assignvariableop_14_conv2d_49_kernel&
"assignvariableop_15_conv2d_49_bias4
0assignvariableop_16_batch_normalization_61_gamma3
/assignvariableop_17_batch_normalization_61_beta:
6assignvariableop_18_batch_normalization_61_moving_mean>
:assignvariableop_19_batch_normalization_61_moving_variance4
0assignvariableop_20_batch_normalization_65_gamma3
/assignvariableop_21_batch_normalization_65_beta:
6assignvariableop_22_batch_normalization_65_moving_mean>
:assignvariableop_23_batch_normalization_65_moving_variance(
$assignvariableop_24_conv2d_47_kernel&
"assignvariableop_25_conv2d_47_bias(
$assignvariableop_26_conv2d_50_kernel&
"assignvariableop_27_conv2d_50_bias4
0assignvariableop_28_batch_normalization_62_gamma3
/assignvariableop_29_batch_normalization_62_beta:
6assignvariableop_30_batch_normalization_62_moving_mean>
:assignvariableop_31_batch_normalization_62_moving_variance4
0assignvariableop_32_batch_normalization_66_gamma3
/assignvariableop_33_batch_normalization_66_beta:
6assignvariableop_34_batch_normalization_66_moving_mean>
:assignvariableop_35_batch_normalization_66_moving_variance'
#assignvariableop_36_dense_30_kernel%
!assignvariableop_37_dense_30_bias'
#assignvariableop_38_dense_32_kernel%
!assignvariableop_39_dense_32_bias4
0assignvariableop_40_batch_normalization_63_gamma3
/assignvariableop_41_batch_normalization_63_beta:
6assignvariableop_42_batch_normalization_63_moving_mean>
:assignvariableop_43_batch_normalization_63_moving_variance4
0assignvariableop_44_batch_normalization_67_gamma3
/assignvariableop_45_batch_normalization_67_beta:
6assignvariableop_46_batch_normalization_67_moving_mean>
:assignvariableop_47_batch_normalization_67_moving_variance'
#assignvariableop_48_dense_31_kernel%
!assignvariableop_49_dense_31_bias'
#assignvariableop_50_dense_33_kernel%
!assignvariableop_51_dense_33_bias!
assignvariableop_52_adam_iter#
assignvariableop_53_adam_beta_1#
assignvariableop_54_adam_beta_2"
assignvariableop_55_adam_decay*
&assignvariableop_56_adam_learning_rate
assignvariableop_57_total
assignvariableop_58_count
assignvariableop_59_total_1
assignvariableop_60_count_1
assignvariableop_61_total_2
assignvariableop_62_count_2
assignvariableop_63_total_3
assignvariableop_64_count_3
assignvariableop_65_total_4
assignvariableop_66_count_4/
+assignvariableop_67_adam_conv2d_45_kernel_m-
)assignvariableop_68_adam_conv2d_45_bias_m/
+assignvariableop_69_adam_conv2d_48_kernel_m-
)assignvariableop_70_adam_conv2d_48_bias_m;
7assignvariableop_71_adam_batch_normalization_60_gamma_m:
6assignvariableop_72_adam_batch_normalization_60_beta_m;
7assignvariableop_73_adam_batch_normalization_64_gamma_m:
6assignvariableop_74_adam_batch_normalization_64_beta_m/
+assignvariableop_75_adam_conv2d_46_kernel_m-
)assignvariableop_76_adam_conv2d_46_bias_m/
+assignvariableop_77_adam_conv2d_49_kernel_m-
)assignvariableop_78_adam_conv2d_49_bias_m;
7assignvariableop_79_adam_batch_normalization_61_gamma_m:
6assignvariableop_80_adam_batch_normalization_61_beta_m;
7assignvariableop_81_adam_batch_normalization_65_gamma_m:
6assignvariableop_82_adam_batch_normalization_65_beta_m/
+assignvariableop_83_adam_conv2d_47_kernel_m-
)assignvariableop_84_adam_conv2d_47_bias_m/
+assignvariableop_85_adam_conv2d_50_kernel_m-
)assignvariableop_86_adam_conv2d_50_bias_m;
7assignvariableop_87_adam_batch_normalization_62_gamma_m:
6assignvariableop_88_adam_batch_normalization_62_beta_m;
7assignvariableop_89_adam_batch_normalization_66_gamma_m:
6assignvariableop_90_adam_batch_normalization_66_beta_m.
*assignvariableop_91_adam_dense_30_kernel_m,
(assignvariableop_92_adam_dense_30_bias_m.
*assignvariableop_93_adam_dense_32_kernel_m,
(assignvariableop_94_adam_dense_32_bias_m;
7assignvariableop_95_adam_batch_normalization_63_gamma_m:
6assignvariableop_96_adam_batch_normalization_63_beta_m;
7assignvariableop_97_adam_batch_normalization_67_gamma_m:
6assignvariableop_98_adam_batch_normalization_67_beta_m.
*assignvariableop_99_adam_dense_31_kernel_m-
)assignvariableop_100_adam_dense_31_bias_m/
+assignvariableop_101_adam_dense_33_kernel_m-
)assignvariableop_102_adam_dense_33_bias_m0
,assignvariableop_103_adam_conv2d_45_kernel_v.
*assignvariableop_104_adam_conv2d_45_bias_v0
,assignvariableop_105_adam_conv2d_48_kernel_v.
*assignvariableop_106_adam_conv2d_48_bias_v<
8assignvariableop_107_adam_batch_normalization_60_gamma_v;
7assignvariableop_108_adam_batch_normalization_60_beta_v<
8assignvariableop_109_adam_batch_normalization_64_gamma_v;
7assignvariableop_110_adam_batch_normalization_64_beta_v0
,assignvariableop_111_adam_conv2d_46_kernel_v.
*assignvariableop_112_adam_conv2d_46_bias_v0
,assignvariableop_113_adam_conv2d_49_kernel_v.
*assignvariableop_114_adam_conv2d_49_bias_v<
8assignvariableop_115_adam_batch_normalization_61_gamma_v;
7assignvariableop_116_adam_batch_normalization_61_beta_v<
8assignvariableop_117_adam_batch_normalization_65_gamma_v;
7assignvariableop_118_adam_batch_normalization_65_beta_v0
,assignvariableop_119_adam_conv2d_47_kernel_v.
*assignvariableop_120_adam_conv2d_47_bias_v0
,assignvariableop_121_adam_conv2d_50_kernel_v.
*assignvariableop_122_adam_conv2d_50_bias_v<
8assignvariableop_123_adam_batch_normalization_62_gamma_v;
7assignvariableop_124_adam_batch_normalization_62_beta_v<
8assignvariableop_125_adam_batch_normalization_66_gamma_v;
7assignvariableop_126_adam_batch_normalization_66_beta_v/
+assignvariableop_127_adam_dense_30_kernel_v-
)assignvariableop_128_adam_dense_30_bias_v/
+assignvariableop_129_adam_dense_32_kernel_v-
)assignvariableop_130_adam_dense_32_bias_v<
8assignvariableop_131_adam_batch_normalization_63_gamma_v;
7assignvariableop_132_adam_batch_normalization_63_beta_v<
8assignvariableop_133_adam_batch_normalization_67_gamma_v;
7assignvariableop_134_adam_batch_normalization_67_beta_v/
+assignvariableop_135_adam_dense_31_kernel_v-
)assignvariableop_136_adam_dense_31_bias_v/
+assignvariableop_137_adam_dense_33_kernel_v-
)assignvariableop_138_adam_dense_33_bias_v
identity_140ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_100вAssignVariableOp_101вAssignVariableOp_102вAssignVariableOp_103вAssignVariableOp_104вAssignVariableOp_105вAssignVariableOp_106вAssignVariableOp_107вAssignVariableOp_108вAssignVariableOp_109вAssignVariableOp_11вAssignVariableOp_110вAssignVariableOp_111вAssignVariableOp_112вAssignVariableOp_113вAssignVariableOp_114вAssignVariableOp_115вAssignVariableOp_116вAssignVariableOp_117вAssignVariableOp_118вAssignVariableOp_119вAssignVariableOp_12вAssignVariableOp_120вAssignVariableOp_121вAssignVariableOp_122вAssignVariableOp_123вAssignVariableOp_124вAssignVariableOp_125вAssignVariableOp_126вAssignVariableOp_127вAssignVariableOp_128вAssignVariableOp_129вAssignVariableOp_13вAssignVariableOp_130вAssignVariableOp_131вAssignVariableOp_132вAssignVariableOp_133вAssignVariableOp_134вAssignVariableOp_135вAssignVariableOp_136вAssignVariableOp_137вAssignVariableOp_138вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_83вAssignVariableOp_84вAssignVariableOp_85вAssignVariableOp_86вAssignVariableOp_87вAssignVariableOp_88вAssignVariableOp_89вAssignVariableOp_9вAssignVariableOp_90вAssignVariableOp_91вAssignVariableOp_92вAssignVariableOp_93вAssignVariableOp_94вAssignVariableOp_95вAssignVariableOp_96вAssignVariableOp_97вAssignVariableOp_98вAssignVariableOp_99╬M
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:М*
dtype0*┘L
value╧LB╠LМB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-14/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesл
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:М*
dtype0*о
valueдBбМB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesю
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╞
_output_shapes│
░::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Э
dtypesТ
П2М	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityа
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_45_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ж
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_45_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2и
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_48_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ж
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_48_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4┤
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_60_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5│
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_60_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6║
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_60_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7╛
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_60_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8┤
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_64_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9│
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_64_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╛
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_64_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11┬
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_64_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12м
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_46_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13к
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_46_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14м
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_49_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15к
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_49_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16╕
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_61_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╖
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_61_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╛
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_61_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19┬
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_61_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20╕
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_65_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╖
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_65_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╛
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_65_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23┬
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_65_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24м
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_47_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25к
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_47_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26м
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_50_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27к
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_50_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╕
AssignVariableOp_28AssignVariableOp0assignvariableop_28_batch_normalization_62_gammaIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╖
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batch_normalization_62_betaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╛
AssignVariableOp_30AssignVariableOp6assignvariableop_30_batch_normalization_62_moving_meanIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31┬
AssignVariableOp_31AssignVariableOp:assignvariableop_31_batch_normalization_62_moving_varianceIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╕
AssignVariableOp_32AssignVariableOp0assignvariableop_32_batch_normalization_66_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33╖
AssignVariableOp_33AssignVariableOp/assignvariableop_33_batch_normalization_66_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╛
AssignVariableOp_34AssignVariableOp6assignvariableop_34_batch_normalization_66_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35┬
AssignVariableOp_35AssignVariableOp:assignvariableop_35_batch_normalization_66_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36л
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_30_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37й
AssignVariableOp_37AssignVariableOp!assignvariableop_37_dense_30_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38л
AssignVariableOp_38AssignVariableOp#assignvariableop_38_dense_32_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39й
AssignVariableOp_39AssignVariableOp!assignvariableop_39_dense_32_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40╕
AssignVariableOp_40AssignVariableOp0assignvariableop_40_batch_normalization_63_gammaIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41╖
AssignVariableOp_41AssignVariableOp/assignvariableop_41_batch_normalization_63_betaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╛
AssignVariableOp_42AssignVariableOp6assignvariableop_42_batch_normalization_63_moving_meanIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43┬
AssignVariableOp_43AssignVariableOp:assignvariableop_43_batch_normalization_63_moving_varianceIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44╕
AssignVariableOp_44AssignVariableOp0assignvariableop_44_batch_normalization_67_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45╖
AssignVariableOp_45AssignVariableOp/assignvariableop_45_batch_normalization_67_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46╛
AssignVariableOp_46AssignVariableOp6assignvariableop_46_batch_normalization_67_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47┬
AssignVariableOp_47AssignVariableOp:assignvariableop_47_batch_normalization_67_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48л
AssignVariableOp_48AssignVariableOp#assignvariableop_48_dense_31_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49й
AssignVariableOp_49AssignVariableOp!assignvariableop_49_dense_31_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50л
AssignVariableOp_50AssignVariableOp#assignvariableop_50_dense_33_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51й
AssignVariableOp_51AssignVariableOp!assignvariableop_51_dense_33_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_52е
AssignVariableOp_52AssignVariableOpassignvariableop_52_adam_iterIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53з
AssignVariableOp_53AssignVariableOpassignvariableop_53_adam_beta_1Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54з
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_beta_2Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55ж
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_decayIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56о
AssignVariableOp_56AssignVariableOp&assignvariableop_56_adam_learning_rateIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57б
AssignVariableOp_57AssignVariableOpassignvariableop_57_totalIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58б
AssignVariableOp_58AssignVariableOpassignvariableop_58_countIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59г
AssignVariableOp_59AssignVariableOpassignvariableop_59_total_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60г
AssignVariableOp_60AssignVariableOpassignvariableop_60_count_1Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61г
AssignVariableOp_61AssignVariableOpassignvariableop_61_total_2Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62г
AssignVariableOp_62AssignVariableOpassignvariableop_62_count_2Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63г
AssignVariableOp_63AssignVariableOpassignvariableop_63_total_3Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64г
AssignVariableOp_64AssignVariableOpassignvariableop_64_count_3Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65г
AssignVariableOp_65AssignVariableOpassignvariableop_65_total_4Identity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66г
AssignVariableOp_66AssignVariableOpassignvariableop_66_count_4Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67│
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv2d_45_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68▒
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv2d_45_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69│
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_48_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70▒
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_48_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71┐
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adam_batch_normalization_60_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72╛
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_60_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73┐
AssignVariableOp_73AssignVariableOp7assignvariableop_73_adam_batch_normalization_64_gamma_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74╛
AssignVariableOp_74AssignVariableOp6assignvariableop_74_adam_batch_normalization_64_beta_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75│
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv2d_46_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76▒
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv2d_46_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77│
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv2d_49_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78▒
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv2d_49_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79┐
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_61_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80╛
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_61_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81┐
AssignVariableOp_81AssignVariableOp7assignvariableop_81_adam_batch_normalization_65_gamma_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82╛
AssignVariableOp_82AssignVariableOp6assignvariableop_82_adam_batch_normalization_65_beta_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83│
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_conv2d_47_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84▒
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adam_conv2d_47_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85│
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_conv2d_50_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86▒
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_conv2d_50_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87┐
AssignVariableOp_87AssignVariableOp7assignvariableop_87_adam_batch_normalization_62_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88╛
AssignVariableOp_88AssignVariableOp6assignvariableop_88_adam_batch_normalization_62_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89┐
AssignVariableOp_89AssignVariableOp7assignvariableop_89_adam_batch_normalization_66_gamma_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90╛
AssignVariableOp_90AssignVariableOp6assignvariableop_90_adam_batch_normalization_66_beta_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91▓
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_dense_30_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92░
AssignVariableOp_92AssignVariableOp(assignvariableop_92_adam_dense_30_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93▓
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_dense_32_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94░
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_dense_32_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95┐
AssignVariableOp_95AssignVariableOp7assignvariableop_95_adam_batch_normalization_63_gamma_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96╛
AssignVariableOp_96AssignVariableOp6assignvariableop_96_adam_batch_normalization_63_beta_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97┐
AssignVariableOp_97AssignVariableOp7assignvariableop_97_adam_batch_normalization_67_gamma_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98╛
AssignVariableOp_98AssignVariableOp6assignvariableop_98_adam_batch_normalization_67_beta_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99▓
AssignVariableOp_99AssignVariableOp*assignvariableop_99_adam_dense_31_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100┤
AssignVariableOp_100AssignVariableOp)assignvariableop_100_adam_dense_31_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101╢
AssignVariableOp_101AssignVariableOp+assignvariableop_101_adam_dense_33_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102┤
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_dense_33_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103╖
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_conv2d_45_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104╡
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_conv2d_45_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105╖
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_conv2d_48_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106╡
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_conv2d_48_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107├
AssignVariableOp_107AssignVariableOp8assignvariableop_107_adam_batch_normalization_60_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108┬
AssignVariableOp_108AssignVariableOp7assignvariableop_108_adam_batch_normalization_60_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109├
AssignVariableOp_109AssignVariableOp8assignvariableop_109_adam_batch_normalization_64_gamma_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110┬
AssignVariableOp_110AssignVariableOp7assignvariableop_110_adam_batch_normalization_64_beta_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111╖
AssignVariableOp_111AssignVariableOp,assignvariableop_111_adam_conv2d_46_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112╡
AssignVariableOp_112AssignVariableOp*assignvariableop_112_adam_conv2d_46_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113╖
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_conv2d_49_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114╡
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_conv2d_49_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115├
AssignVariableOp_115AssignVariableOp8assignvariableop_115_adam_batch_normalization_61_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116┬
AssignVariableOp_116AssignVariableOp7assignvariableop_116_adam_batch_normalization_61_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117├
AssignVariableOp_117AssignVariableOp8assignvariableop_117_adam_batch_normalization_65_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118┬
AssignVariableOp_118AssignVariableOp7assignvariableop_118_adam_batch_normalization_65_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119╖
AssignVariableOp_119AssignVariableOp,assignvariableop_119_adam_conv2d_47_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120╡
AssignVariableOp_120AssignVariableOp*assignvariableop_120_adam_conv2d_47_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121╖
AssignVariableOp_121AssignVariableOp,assignvariableop_121_adam_conv2d_50_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122╡
AssignVariableOp_122AssignVariableOp*assignvariableop_122_adam_conv2d_50_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123├
AssignVariableOp_123AssignVariableOp8assignvariableop_123_adam_batch_normalization_62_gamma_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124┬
AssignVariableOp_124AssignVariableOp7assignvariableop_124_adam_batch_normalization_62_beta_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125├
AssignVariableOp_125AssignVariableOp8assignvariableop_125_adam_batch_normalization_66_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126┬
AssignVariableOp_126AssignVariableOp7assignvariableop_126_adam_batch_normalization_66_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127╢
AssignVariableOp_127AssignVariableOp+assignvariableop_127_adam_dense_30_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128┤
AssignVariableOp_128AssignVariableOp)assignvariableop_128_adam_dense_30_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129╢
AssignVariableOp_129AssignVariableOp+assignvariableop_129_adam_dense_32_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130┤
AssignVariableOp_130AssignVariableOp)assignvariableop_130_adam_dense_32_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131├
AssignVariableOp_131AssignVariableOp8assignvariableop_131_adam_batch_normalization_63_gamma_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132┬
AssignVariableOp_132AssignVariableOp7assignvariableop_132_adam_batch_normalization_63_beta_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133├
AssignVariableOp_133AssignVariableOp8assignvariableop_133_adam_batch_normalization_67_gamma_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134┬
AssignVariableOp_134AssignVariableOp7assignvariableop_134_adam_batch_normalization_67_beta_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135╢
AssignVariableOp_135AssignVariableOp+assignvariableop_135_adam_dense_31_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136┤
AssignVariableOp_136AssignVariableOp)assignvariableop_136_adam_dense_31_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137╢
AssignVariableOp_137AssignVariableOp+assignvariableop_137_adam_dense_33_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138┤
AssignVariableOp_138AssignVariableOp)assignvariableop_138_adam_dense_33_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp∙
Identity_139Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_139э
Identity_140IdentityIdentity_139:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_140"%
identity_140Identity_140:output:0*├
_input_shapes▒
о: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
░
L
0__inference_max_pooling2d_49_layer_call_fn_38302

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_382962
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ш
c
E__inference_dropout_61_layer_call_and_return_conditional_losses_42231

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         !! 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         !! 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
Ф	
▄
C__inference_dense_31_layer_call_and_return_conditional_losses_42944

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
р
й
6__inference_batch_normalization_64_layer_call_fn_41769

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_389152
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
┴
Ї
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_38499

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ш
c
E__inference_dropout_66_layer_call_and_return_conditional_losses_39633

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:          2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
∙
Ї
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_42060

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
╓
d
H__inference_activation_66_layer_call_and_return_conditional_losses_42321

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         !! 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
ш
c
E__inference_dropout_66_layer_call_and_return_conditional_losses_42626

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:          2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╜
a
E__inference_flatten_15_layer_call_and_return_conditional_losses_39696

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"        2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
в
й
6__inference_batch_normalization_64_layer_call_fn_41846

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_380352
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╜
I
-__inference_activation_65_layer_call_fn_41958

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_65_layer_call_and_return_conditional_losses_391612
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB :W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
в
й
6__inference_batch_normalization_65_layer_call_fn_42150

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_382672
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Б
Ї
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41628

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╠
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3▄
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
а
й
6__inference_batch_normalization_64_layer_call_fn_41833

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_380042
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┴
Ї
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42428

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
░
L
0__inference_max_pooling2d_47_layer_call_fn_38522

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_385162
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
├
d
E__inference_dropout_62_layer_call_and_return_conditional_losses_39658

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
∙
Ї
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42188

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         BB : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
∙
Ї
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_39505

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
в
й
6__inference_batch_normalization_60_layer_call_fn_41718

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_379312
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ш
c
E__inference_dropout_65_layer_call_and_return_conditional_losses_39347

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         !! 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         !! 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
┌
й
6__inference_batch_normalization_66_layer_call_fn_42582

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_395052
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
ц
е
(__inference_face_net_layer_call_fn_40673
input_6
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity

identity_1ИвStatefulPartitionedCall╛
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_face_net_layer_call_and_return_conditional_losses_405642
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ╞╞
!
_user_specified_name	input_6
┴
Ї
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_38395

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_38004

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
║
d
H__inference_activation_63_layer_call_and_return_conditional_losses_42701

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╖
F
*__inference_dropout_65_layer_call_fn_42268

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_65_layer_call_and_return_conditional_losses_393472
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
├
d
E__inference_dropout_60_layer_call_and_return_conditional_losses_41858

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         BB2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         BB*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         BB2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         BB2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         BB2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         BB2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
С0
╞
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_38770

inputs
assignmovingavg_38745
assignmovingavg_1_38751)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/38745*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_38745*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/38745*
_output_shapes	
:А2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/38745*
_output_shapes	
:А2
AssignMovingAvg/mulн
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_38745AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/38745*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╤
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/38751*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_38751*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp√
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/38751*
_output_shapes	
:А2
AssignMovingAvg_1/subЄ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/38751*
_output_shapes	
:А2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_38751AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/38751*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
а
й
6__inference_batch_normalization_61_layer_call_fn_42009

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_381322
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42106

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ъ	
▌
D__inference_conv2d_50_layer_call_and_return_conditional_losses_39400

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         !! 2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         !! ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
в
й
6__inference_batch_normalization_62_layer_call_fn_42454

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_383952
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
├
d
E__inference_dropout_61_layer_call_and_return_conditional_losses_39372

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         !! 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         !! *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         !! 2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         !! 2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         !! 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
С0
╞
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_42752

inputs
assignmovingavg_42727
assignmovingavg_1_42733)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices│
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/42727*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayУ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_42727*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/42727*
_output_shapes	
:А2
AssignMovingAvg/subш
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*(
_class
loc:@AssignMovingAvg/42727*
_output_shapes	
:А2
AssignMovingAvg/mulн
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_42727AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*(
_class
loc:@AssignMovingAvg/42727*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╤
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/42733*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЩ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_42733*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOp√
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/42733*
_output_shapes	
:А2
AssignMovingAvg_1/subЄ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/42733*
_output_shapes	
:А2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_42733AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0**
_class 
loc:@AssignMovingAvg_1/42733*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1┤
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ъ	
▌
D__inference_conv2d_49_layer_call_and_return_conditional_losses_39114

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpг
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         BB 2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         BB::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
┴
Ї
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42124

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
б
I
-__inference_activation_63_layer_call_fn_42706

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_63_layer_call_and_return_conditional_losses_397742
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
║
d
H__inference_activation_67_layer_call_and_return_conditional_losses_42711

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:         А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
├
c
*__inference_dropout_64_layer_call_fn_41895

inputs
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_64_layer_call_and_return_conditional_losses_390562
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         BB2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
Ф	
▄
C__inference_dense_33_layer_call_and_return_conditional_losses_39922

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╓
d
H__inference_activation_62_layer_call_and_return_conditional_losses_42311

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         !! 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
░
L
0__inference_max_pooling2d_50_layer_call_fn_38534

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_385282
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Е
Ш
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_39560

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
ш
c
E__inference_dropout_60_layer_call_and_return_conditional_losses_39091

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         BB2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         BB2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         BB:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
∙
Ї
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_39578

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
╫
a
E__inference_age_output_layer_call_and_return_conditional_losses_42976

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ф	
▄
C__inference_dense_31_layer_call_and_return_conditional_losses_39948

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
├
d
E__inference_dropout_61_layer_call_and_return_conditional_losses_42226

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         !! 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         !! *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         !! 2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         !! 2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         !! 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
┴
Ї
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_38035

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
а
й
6__inference_batch_normalization_65_layer_call_fn_42137

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_382362
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╓
d
H__inference_activation_61_layer_call_and_return_conditional_losses_39174

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         BB 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB :W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
т
й
6__inference_batch_normalization_60_layer_call_fn_41654

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_390062
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
ш
c
E__inference_dropout_65_layer_call_and_return_conditional_losses_42258

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         !! 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         !! 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
Н
Ш
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_38915

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1А
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
 
~
)__inference_conv2d_46_layer_call_fn_41919

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_46_layer_call_and_return_conditional_losses_391402
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         BB::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_37900

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
т
И
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_38663

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┴
Ї
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42492

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
т
И
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_38803

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2
batchnorm/add_1▄
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ї	
▌
D__inference_conv2d_45_layer_call_and_return_conditional_losses_41542

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpе
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╞╞2	
BiasAddЯ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╞╞::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41674

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Щ	
▄
C__inference_dense_30_layer_call_and_return_conditional_losses_39740

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
├
d
E__inference_dropout_66_layer_call_and_return_conditional_losses_39628

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
├╟
╗
C__inference_face_net_layer_call_and_return_conditional_losses_40144
input_6
conv2d_48_39994
conv2d_48_39996
conv2d_45_39999
conv2d_45_40001 
batch_normalization_64_40006 
batch_normalization_64_40008 
batch_normalization_64_40010 
batch_normalization_64_40012 
batch_normalization_60_40015 
batch_normalization_60_40017 
batch_normalization_60_40019 
batch_normalization_60_40021
conv2d_49_40028
conv2d_49_40030
conv2d_46_40033
conv2d_46_40035 
batch_normalization_65_40040 
batch_normalization_65_40042 
batch_normalization_65_40044 
batch_normalization_65_40046 
batch_normalization_61_40049 
batch_normalization_61_40051 
batch_normalization_61_40053 
batch_normalization_61_40055
conv2d_50_40062
conv2d_50_40064
conv2d_47_40067
conv2d_47_40069 
batch_normalization_66_40074 
batch_normalization_66_40076 
batch_normalization_66_40078 
batch_normalization_66_40080 
batch_normalization_62_40083 
batch_normalization_62_40085 
batch_normalization_62_40087 
batch_normalization_62_40089
dense_32_40098
dense_32_40100
dense_30_40103
dense_30_40105 
batch_normalization_67_40110 
batch_normalization_67_40112 
batch_normalization_67_40114 
batch_normalization_67_40116 
batch_normalization_63_40119 
batch_normalization_63_40121 
batch_normalization_63_40123 
batch_normalization_63_40125
dense_33_40130
dense_33_40132
dense_31_40135
dense_31_40137
identity

identity_1Ив.batch_normalization_60/StatefulPartitionedCallв.batch_normalization_61/StatefulPartitionedCallв.batch_normalization_62/StatefulPartitionedCallв.batch_normalization_63/StatefulPartitionedCallв.batch_normalization_64/StatefulPartitionedCallв.batch_normalization_65/StatefulPartitionedCallв.batch_normalization_66/StatefulPartitionedCallв.batch_normalization_67/StatefulPartitionedCallв!conv2d_45/StatefulPartitionedCallв!conv2d_46/StatefulPartitionedCallв!conv2d_47/StatefulPartitionedCallв!conv2d_48/StatefulPartitionedCallв!conv2d_49/StatefulPartitionedCallв!conv2d_50/StatefulPartitionedCallв dense_30/StatefulPartitionedCallв dense_31/StatefulPartitionedCallв dense_32/StatefulPartitionedCallв dense_33/StatefulPartitionedCallд
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCallinput_6conv2d_48_39994conv2d_48_39996*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_48_layer_call_and_return_conditional_losses_388282#
!conv2d_48/StatefulPartitionedCallд
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCallinput_6conv2d_45_39999conv2d_45_40001*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_388542#
!conv2d_45/StatefulPartitionedCallУ
activation_64/PartitionedCallPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_64_layer_call_and_return_conditional_losses_388752
activation_64/PartitionedCallУ
activation_60/PartitionedCallPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_60_layer_call_and_return_conditional_losses_388882
activation_60/PartitionedCall─
.batch_normalization_64/StatefulPartitionedCallStatefulPartitionedCall&activation_64/PartitionedCall:output:0batch_normalization_64_40006batch_normalization_64_40008batch_normalization_64_40010batch_normalization_64_40012*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_3893320
.batch_normalization_64/StatefulPartitionedCall─
.batch_normalization_60/StatefulPartitionedCallStatefulPartitionedCall&activation_60/PartitionedCall:output:0batch_normalization_60_40015batch_normalization_60_40017batch_normalization_60_40019batch_normalization_60_40021*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_3900620
.batch_normalization_60/StatefulPartitionedCallз
 max_pooling2d_48/PartitionedCallPartitionedCall7batch_normalization_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_380642"
 max_pooling2d_48/PartitionedCallз
 max_pooling2d_45/PartitionedCallPartitionedCall7batch_normalization_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_380522"
 max_pooling2d_45/PartitionedCallЗ
dropout_64/PartitionedCallPartitionedCall)max_pooling2d_48/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_64_layer_call_and_return_conditional_losses_390612
dropout_64/PartitionedCallЗ
dropout_60/PartitionedCallPartitionedCall)max_pooling2d_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_60_layer_call_and_return_conditional_losses_390912
dropout_60/PartitionedCall╛
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall#dropout_64/PartitionedCall:output:0conv2d_49_40028conv2d_49_40030*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_391142#
!conv2d_49/StatefulPartitionedCall╛
!conv2d_46/StatefulPartitionedCallStatefulPartitionedCall#dropout_60/PartitionedCall:output:0conv2d_46_40033conv2d_46_40035*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_46_layer_call_and_return_conditional_losses_391402#
!conv2d_46/StatefulPartitionedCallС
activation_65/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_65_layer_call_and_return_conditional_losses_391612
activation_65/PartitionedCallС
activation_61/PartitionedCallPartitionedCall*conv2d_46/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_61_layer_call_and_return_conditional_losses_391742
activation_61/PartitionedCall┬
.batch_normalization_65/StatefulPartitionedCallStatefulPartitionedCall&activation_65/PartitionedCall:output:0batch_normalization_65_40040batch_normalization_65_40042batch_normalization_65_40044batch_normalization_65_40046*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_3921920
.batch_normalization_65/StatefulPartitionedCall┬
.batch_normalization_61/StatefulPartitionedCallStatefulPartitionedCall&activation_61/PartitionedCall:output:0batch_normalization_61_40049batch_normalization_61_40051batch_normalization_61_40053batch_normalization_61_40055*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_3929220
.batch_normalization_61/StatefulPartitionedCallз
 max_pooling2d_49/PartitionedCallPartitionedCall7batch_normalization_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_382962"
 max_pooling2d_49/PartitionedCallз
 max_pooling2d_46/PartitionedCallPartitionedCall7batch_normalization_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_382842"
 max_pooling2d_46/PartitionedCallЗ
dropout_65/PartitionedCallPartitionedCall)max_pooling2d_49/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_65_layer_call_and_return_conditional_losses_393472
dropout_65/PartitionedCallЗ
dropout_61/PartitionedCallPartitionedCall)max_pooling2d_46/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_61_layer_call_and_return_conditional_losses_393772
dropout_61/PartitionedCall╛
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall#dropout_65/PartitionedCall:output:0conv2d_50_40062conv2d_50_40064*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_394002#
!conv2d_50/StatefulPartitionedCall╛
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall#dropout_61/PartitionedCall:output:0conv2d_47_40067conv2d_47_40069*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_394262#
!conv2d_47/StatefulPartitionedCallС
activation_66/PartitionedCallPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_66_layer_call_and_return_conditional_losses_394472
activation_66/PartitionedCallС
activation_62/PartitionedCallPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_62_layer_call_and_return_conditional_losses_394602
activation_62/PartitionedCall┬
.batch_normalization_66/StatefulPartitionedCallStatefulPartitionedCall&activation_66/PartitionedCall:output:0batch_normalization_66_40074batch_normalization_66_40076batch_normalization_66_40078batch_normalization_66_40080*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_3950520
.batch_normalization_66/StatefulPartitionedCall┬
.batch_normalization_62/StatefulPartitionedCallStatefulPartitionedCall&activation_62/PartitionedCall:output:0batch_normalization_62_40083batch_normalization_62_40085batch_normalization_62_40087batch_normalization_62_40089*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_3957820
.batch_normalization_62/StatefulPartitionedCallз
 max_pooling2d_50/PartitionedCallPartitionedCall7batch_normalization_66/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_385282"
 max_pooling2d_50/PartitionedCallз
 max_pooling2d_47/PartitionedCallPartitionedCall7batch_normalization_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_385162"
 max_pooling2d_47/PartitionedCallЗ
dropout_66/PartitionedCallPartitionedCall)max_pooling2d_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_66_layer_call_and_return_conditional_losses_396332
dropout_66/PartitionedCallЗ
dropout_62/PartitionedCallPartitionedCall)max_pooling2d_47/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_62_layer_call_and_return_conditional_losses_396632
dropout_62/PartitionedCall·
flatten_16/PartitionedCallPartitionedCall#dropout_66/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_396822
flatten_16/PartitionedCall·
flatten_15/PartitionedCallPartitionedCall#dropout_62/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_flatten_15_layer_call_and_return_conditional_losses_396962
flatten_15/PartitionedCall▓
 dense_32/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_32_40098dense_32_40100*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_397142"
 dense_32/StatefulPartitionedCall▓
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_30_40103dense_30_40105*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_397402"
 dense_30/StatefulPartitionedCallЙ
activation_67/PartitionedCallPartitionedCall)dense_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_67_layer_call_and_return_conditional_losses_397612
activation_67/PartitionedCallЙ
activation_63/PartitionedCallPartitionedCall)dense_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_63_layer_call_and_return_conditional_losses_397742
activation_63/PartitionedCall╗
.batch_normalization_67/StatefulPartitionedCallStatefulPartitionedCall&activation_67/PartitionedCall:output:0batch_normalization_67_40110batch_normalization_67_40112batch_normalization_67_40114batch_normalization_67_40116*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_3880320
.batch_normalization_67/StatefulPartitionedCall╗
.batch_normalization_63/StatefulPartitionedCallStatefulPartitionedCall&activation_63/PartitionedCall:output:0batch_normalization_63_40119batch_normalization_63_40121batch_normalization_63_40123batch_normalization_63_40125*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_3866320
.batch_normalization_63/StatefulPartitionedCallО
dropout_67/PartitionedCallPartitionedCall7batch_normalization_67/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_67_layer_call_and_return_conditional_losses_398692
dropout_67/PartitionedCallО
dropout_63/PartitionedCallPartitionedCall7batch_normalization_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_63_layer_call_and_return_conditional_losses_398992
dropout_63/PartitionedCall▒
 dense_33/StatefulPartitionedCallStatefulPartitionedCall#dropout_67/PartitionedCall:output:0dense_33_40130dense_33_40132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_33_layer_call_and_return_conditional_losses_399222"
 dense_33/StatefulPartitionedCall▒
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#dropout_63/PartitionedCall:output:0dense_31_40135dense_31_40137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_399482"
 dense_31/StatefulPartitionedCallИ
gender_output/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_gender_output_layer_call_and_return_conditional_losses_399692
gender_output/PartitionedCall 
age_output/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_age_output_layer_call_and_return_conditional_losses_399812
age_output/PartitionedCallу
IdentityIdentity#age_output/PartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identityъ

Identity_1Identity&gender_output/PartitionedCall:output:0/^batch_normalization_60/StatefulPartitionedCall/^batch_normalization_61/StatefulPartitionedCall/^batch_normalization_62/StatefulPartitionedCall/^batch_normalization_63/StatefulPartitionedCall/^batch_normalization_64/StatefulPartitionedCall/^batch_normalization_65/StatefulPartitionedCall/^batch_normalization_66/StatefulPartitionedCall/^batch_normalization_67/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_46/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::2`
.batch_normalization_60/StatefulPartitionedCall.batch_normalization_60/StatefulPartitionedCall2`
.batch_normalization_61/StatefulPartitionedCall.batch_normalization_61/StatefulPartitionedCall2`
.batch_normalization_62/StatefulPartitionedCall.batch_normalization_62/StatefulPartitionedCall2`
.batch_normalization_63/StatefulPartitionedCall.batch_normalization_63/StatefulPartitionedCall2`
.batch_normalization_64/StatefulPartitionedCall.batch_normalization_64/StatefulPartitionedCall2`
.batch_normalization_65/StatefulPartitionedCall.batch_normalization_65/StatefulPartitionedCall2`
.batch_normalization_66/StatefulPartitionedCall.batch_normalization_66/StatefulPartitionedCall2`
.batch_normalization_67/StatefulPartitionedCall.batch_normalization_67/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_46/StatefulPartitionedCall!conv2d_46/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:Z V
1
_output_shapes
:         ╞╞
!
_user_specified_name	input_6
├
c
*__inference_dropout_65_layer_call_fn_42263

inputs
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_65_layer_call_and_return_conditional_losses_393422
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
 
~
)__inference_conv2d_49_layer_call_fn_41938

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_391142
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         BB::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
Н
Ш
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41610

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1┌
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:         ╞╞:::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1А
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:         ╞╞::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
Ч
F
*__inference_age_output_layer_call_fn_42981

inputs
identity╞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_age_output_layer_call_and_return_conditional_losses_399812
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
с
}
(__inference_dense_32_layer_call_fn_42696

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_32_layer_call_and_return_conditional_losses_397142
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
╓
d
H__inference_activation_65_layer_call_and_return_conditional_losses_39161

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         BB 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB :W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
ш
c
E__inference_dropout_61_layer_call_and_return_conditional_losses_39377

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         !! 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         !! 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42410

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
┼
I
-__inference_activation_64_layer_call_fn_41590

inputs
identity╙
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ╞╞* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_64_layer_call_and_return_conditional_losses_388752
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ╞╞2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ╞╞:Y U
1
_output_shapes
:         ╞╞
 
_user_specified_nameinputs
╛
й
6__inference_batch_normalization_63_layer_call_fn_42798

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_386632
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╪
й
6__inference_batch_normalization_62_layer_call_fn_42377

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_395602
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
┌
й
6__inference_batch_normalization_61_layer_call_fn_42086

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         BB *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_392922
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         BB 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         BB ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         BB 
 
_user_specified_nameinputs
├
c
*__inference_dropout_66_layer_call_fn_42631

inputs
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_66_layer_call_and_return_conditional_losses_396282
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╖
F
*__inference_dropout_66_layer_call_fn_42636

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_66_layer_call_and_return_conditional_losses_396332
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
с
}
(__inference_dense_30_layer_call_fn_42677

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_397402
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А@::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42474

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
а
й
6__inference_batch_normalization_60_layer_call_fn_41705

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_379002
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
∙
Ї
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42556

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
╛
а
#__inference_signature_wrapper_40794
input_6
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50
identity

identity_1ИвStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         :         *V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./01234*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_378382
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*В
_input_shapesЁ
э:         ╞╞::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ╞╞
!
_user_specified_name	input_6
з
c
*__inference_dropout_63_layer_call_fn_42902

inputs
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_63_layer_call_and_return_conditional_losses_398942
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б
I
-__inference_activation_67_layer_call_fn_42716

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_67_layer_call_and_return_conditional_losses_397612
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
∙
Ї
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42364

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         !! : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         !! ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_38296

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
а
й
6__inference_batch_normalization_62_layer_call_fn_42441

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╢
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_383642
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41802

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Б
g
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_38516

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
├
d
E__inference_dropout_64_layer_call_and_return_conditional_losses_41885

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         BB2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         BB*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         BB2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         BB2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         BB2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         BB2

Identity"
identityIdentity:output:0*.
_input_shapes
:         BB:W S
/
_output_shapes
:         BB
 
_user_specified_nameinputs
╝
й
6__inference_batch_normalization_63_layer_call_fn_42785

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_386302
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
═
Ш
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41978

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╜
I
-__inference_activation_66_layer_call_fn_42326

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_activation_66_layer_call_and_return_conditional_losses_394472
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! :W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
├
c
*__inference_dropout_61_layer_call_fn_42236

inputs
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dropout_61_layer_call_and_return_conditional_losses_393722
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*.
_input_shapes
:         !! 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
 
~
)__inference_conv2d_50_layer_call_fn_42306

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         !! *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_394002
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         !! 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         !! ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         !! 
 
_user_specified_nameinputs
╠
c
E__inference_dropout_67_layer_call_and_return_conditional_losses_39869

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*·
serving_defaultц
E
input_6:
serving_default_input_6:0         ╞╞>

age_output0
StatefulPartitionedCall:0         A
gender_output0
StatefulPartitionedCall:1         tensorflow/serving/predict:┤е

Ъ│
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
layer-17
layer-18
layer-19
layer-20
layer_with_weights-8
layer-21
layer_with_weights-9
layer-22
layer-23
layer-24
layer_with_weights-10
layer-25
layer_with_weights-11
layer-26
layer-27
layer-28
layer-29
layer-30
 layer-31
!layer-32
"layer_with_weights-12
"layer-33
#layer_with_weights-13
#layer-34
$layer-35
%layer-36
&layer_with_weights-14
&layer-37
'layer_with_weights-15
'layer-38
(layer-39
)layer-40
*layer_with_weights-16
*layer-41
+layer_with_weights-17
+layer-42
,layer-43
-layer-44
.	optimizer
/loss
0regularization_losses
1trainable_variables
2	variables
3	keras_api
4
signatures
ъ__call__
+ы&call_and_return_all_conditional_losses
ь_default_save_signature"∙и
_tf_keras_network▄и{"class_name": "Functional", "name": "face_net", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "face_net", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 198, 198, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_45", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_45", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_48", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_48", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_60", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_60", "inbound_nodes": [[["conv2d_45", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_64", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_64", "inbound_nodes": [[["conv2d_48", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_60", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_60", "inbound_nodes": [[["activation_60", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_64", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_64", "inbound_nodes": [[["activation_64", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_45", "inbound_nodes": [[["batch_normalization_60", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_48", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_48", "inbound_nodes": [[["batch_normalization_64", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_60", "inbound_nodes": [[["max_pooling2d_45", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_64", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_64", "inbound_nodes": [[["max_pooling2d_48", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_46", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_46", "inbound_nodes": [[["dropout_60", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_49", "inbound_nodes": [[["dropout_64", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_61", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_61", "inbound_nodes": [[["conv2d_46", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_65", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_65", "inbound_nodes": [[["conv2d_49", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_61", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_61", "inbound_nodes": [[["activation_61", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_65", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_65", "inbound_nodes": [[["activation_65", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_46", "inbound_nodes": [[["batch_normalization_61", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_49", "inbound_nodes": [[["batch_normalization_65", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_61", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_61", "inbound_nodes": [[["max_pooling2d_46", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_65", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_65", "inbound_nodes": [[["max_pooling2d_49", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_47", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_47", "inbound_nodes": [[["dropout_61", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_50", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_50", "inbound_nodes": [[["dropout_65", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_62", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_62", "inbound_nodes": [[["conv2d_47", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_66", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_66", "inbound_nodes": [[["conv2d_50", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_62", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_62", "inbound_nodes": [[["activation_62", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_66", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_66", "inbound_nodes": [[["activation_66", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_47", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_47", "inbound_nodes": [[["batch_normalization_62", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_50", "inbound_nodes": [[["batch_normalization_66", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_62", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_62", "inbound_nodes": [[["max_pooling2d_47", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_66", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_66", "inbound_nodes": [[["max_pooling2d_50", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_15", "inbound_nodes": [[["dropout_62", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_16", "inbound_nodes": [[["dropout_66", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_30", "inbound_nodes": [[["flatten_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["flatten_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_63", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_63", "inbound_nodes": [[["dense_30", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_67", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_67", "inbound_nodes": [[["dense_32", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_63", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_63", "inbound_nodes": [[["activation_63", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_67", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_67", "inbound_nodes": [[["activation_67", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_63", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_63", "inbound_nodes": [[["batch_normalization_63", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_67", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_67", "inbound_nodes": [[["batch_normalization_67", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_31", "inbound_nodes": [[["dropout_63", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["dropout_67", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "age_output", "trainable": true, "dtype": "float32", "activation": "linear"}, "name": "age_output", "inbound_nodes": [[["dense_31", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "gender_output", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "gender_output", "inbound_nodes": [[["dense_33", 0, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["age_output", 0, 0], ["gender_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 198, 198, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 198, 198, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "face_net", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 198, 198, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_45", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_45", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_48", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_48", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_60", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_60", "inbound_nodes": [[["conv2d_45", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_64", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_64", "inbound_nodes": [[["conv2d_48", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_60", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_60", "inbound_nodes": [[["activation_60", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_64", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_64", "inbound_nodes": [[["activation_64", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_45", "inbound_nodes": [[["batch_normalization_60", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_48", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "name": "max_pooling2d_48", "inbound_nodes": [[["batch_normalization_64", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_60", "inbound_nodes": [[["max_pooling2d_45", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_64", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_64", "inbound_nodes": [[["max_pooling2d_48", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_46", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_46", "inbound_nodes": [[["dropout_60", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_49", "inbound_nodes": [[["dropout_64", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_61", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_61", "inbound_nodes": [[["conv2d_46", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_65", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_65", "inbound_nodes": [[["conv2d_49", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_61", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_61", "inbound_nodes": [[["activation_61", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_65", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_65", "inbound_nodes": [[["activation_65", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_46", "inbound_nodes": [[["batch_normalization_61", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_49", "inbound_nodes": [[["batch_normalization_65", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_61", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_61", "inbound_nodes": [[["max_pooling2d_46", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_65", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_65", "inbound_nodes": [[["max_pooling2d_49", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_47", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_47", "inbound_nodes": [[["dropout_61", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_50", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_50", "inbound_nodes": [[["dropout_65", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_62", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_62", "inbound_nodes": [[["conv2d_47", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_66", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_66", "inbound_nodes": [[["conv2d_50", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_62", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_62", "inbound_nodes": [[["activation_62", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_66", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_66", "inbound_nodes": [[["activation_66", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_47", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_47", "inbound_nodes": [[["batch_normalization_62", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_50", "inbound_nodes": [[["batch_normalization_66", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_62", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_62", "inbound_nodes": [[["max_pooling2d_47", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_66", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "name": "dropout_66", "inbound_nodes": [[["max_pooling2d_50", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_15", "inbound_nodes": [[["dropout_62", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_16", "inbound_nodes": [[["dropout_66", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_30", "inbound_nodes": [[["flatten_15", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_32", "inbound_nodes": [[["flatten_16", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_63", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_63", "inbound_nodes": [[["dense_30", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_67", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_67", "inbound_nodes": [[["dense_32", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_63", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_63", "inbound_nodes": [[["activation_63", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_67", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_67", "inbound_nodes": [[["activation_67", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_63", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_63", "inbound_nodes": [[["batch_normalization_63", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_67", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_67", "inbound_nodes": [[["batch_normalization_67", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_31", "inbound_nodes": [[["dropout_63", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_33", "inbound_nodes": [[["dropout_67", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "age_output", "trainable": true, "dtype": "float32", "activation": "linear"}, "name": "age_output", "inbound_nodes": [[["dense_31", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "gender_output", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "gender_output", "inbound_nodes": [[["dense_33", 0, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["age_output", 0, 0], ["gender_output", 0, 0]]}}, "training_config": {"loss": {"age_output": "mse", "gender_output": "binary_crossentropy"}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "age_output_mae", "dtype": "float32", "fn": "mean_absolute_error"}}], [{"class_name": "MeanMetricWrapper", "config": {"name": "gender_output_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": {"age_output": 4.0, "gender_output": 0.1}, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 9.999999747378752e-05, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
¤"·
_tf_keras_input_layer┌{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 198, 198, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 198, 198, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
°	

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
э__call__
+ю&call_and_return_all_conditional_losses"╤
_tf_keras_layer╖{"class_name": "Conv2D", "name": "conv2d_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_45", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 198, 198, 3]}}
°	

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
я__call__
+Ё&call_and_return_all_conditional_losses"╤
_tf_keras_layer╖{"class_name": "Conv2D", "name": "conv2d_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_48", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 198, 198, 3]}}
┘
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
ё__call__
+Є&call_and_return_all_conditional_losses"╚
_tf_keras_layerо{"class_name": "Activation", "name": "activation_60", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_60", "trainable": true, "dtype": "float32", "activation": "relu"}}
┘
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses"╚
_tf_keras_layerо{"class_name": "Activation", "name": "activation_64", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_64", "trainable": true, "dtype": "float32", "activation": "relu"}}
└	
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
ї__call__
+Ў&call_and_return_all_conditional_losses"ъ
_tf_keras_layer╨{"class_name": "BatchNormalization", "name": "batch_normalization_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_60", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 198, 198, 16]}}
└	
Raxis
	Sgamma
Tbeta
Umoving_mean
Vmoving_variance
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
ў__call__
+°&call_and_return_all_conditional_losses"ъ
_tf_keras_layer╨{"class_name": "BatchNormalization", "name": "batch_normalization_64", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_64", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 198, 198, 16]}}
Г
[regularization_losses
\trainable_variables
]	variables
^	keras_api
∙__call__
+·&call_and_return_all_conditional_losses"Є
_tf_keras_layer╪{"class_name": "MaxPooling2D", "name": "max_pooling2d_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Г
_regularization_losses
`trainable_variables
a	variables
b	keras_api
√__call__
+№&call_and_return_all_conditional_losses"Є
_tf_keras_layer╪{"class_name": "MaxPooling2D", "name": "max_pooling2d_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_48", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [3, 3]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ъ
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
¤__call__
+■&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"class_name": "Dropout", "name": "dropout_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_60", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
ъ
gregularization_losses
htrainable_variables
i	variables
j	keras_api
 __call__
+А&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"class_name": "Dropout", "name": "dropout_64", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_64", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
°	

kkernel
lbias
mregularization_losses
ntrainable_variables
o	variables
p	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"╤
_tf_keras_layer╖{"class_name": "Conv2D", "name": "conv2d_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_46", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 66, 16]}}
°	

qkernel
rbias
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"╤
_tf_keras_layer╖{"class_name": "Conv2D", "name": "conv2d_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 66, 16]}}
┘
wregularization_losses
xtrainable_variables
y	variables
z	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"╚
_tf_keras_layerо{"class_name": "Activation", "name": "activation_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_61", "trainable": true, "dtype": "float32", "activation": "relu"}}
┘
{regularization_losses
|trainable_variables
}	variables
~	keras_api
З__call__
+И&call_and_return_all_conditional_losses"╚
_tf_keras_layerо{"class_name": "Activation", "name": "activation_65", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_65", "trainable": true, "dtype": "float32", "activation": "relu"}}
╞	
axis

Аgamma
	Бbeta
Вmoving_mean
Гmoving_variance
Дregularization_losses
Еtrainable_variables
Ж	variables
З	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"ш
_tf_keras_layer╬{"class_name": "BatchNormalization", "name": "batch_normalization_61", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_61", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 66, 32]}}
╟	
	Иaxis

Йgamma
	Кbeta
Лmoving_mean
Мmoving_variance
Нregularization_losses
Оtrainable_variables
П	variables
Р	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"ш
_tf_keras_layer╬{"class_name": "BatchNormalization", "name": "batch_normalization_65", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_65", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 66, 66, 32]}}
З
Сregularization_losses
Тtrainable_variables
У	variables
Ф	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"Є
_tf_keras_layer╪{"class_name": "MaxPooling2D", "name": "max_pooling2d_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
З
Хregularization_losses
Цtrainable_variables
Ч	variables
Ш	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"Є
_tf_keras_layer╪{"class_name": "MaxPooling2D", "name": "max_pooling2d_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_49", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ю
Щregularization_losses
Ъtrainable_variables
Ы	variables
Ь	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"class_name": "Dropout", "name": "dropout_61", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_61", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
ю
Эregularization_losses
Юtrainable_variables
Я	variables
а	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"class_name": "Dropout", "name": "dropout_65", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_65", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
■	
бkernel
	вbias
гregularization_losses
дtrainable_variables
е	variables
ж	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"╤
_tf_keras_layer╖{"class_name": "Conv2D", "name": "conv2d_47", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_47", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 32]}}
■	
зkernel
	иbias
йregularization_losses
кtrainable_variables
л	variables
м	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"╤
_tf_keras_layer╖{"class_name": "Conv2D", "name": "conv2d_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_50", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 32]}}
▌
нregularization_losses
оtrainable_variables
п	variables
░	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"╚
_tf_keras_layerо{"class_name": "Activation", "name": "activation_62", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_62", "trainable": true, "dtype": "float32", "activation": "relu"}}
▌
▒regularization_losses
▓trainable_variables
│	variables
┤	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"╚
_tf_keras_layerо{"class_name": "Activation", "name": "activation_66", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_66", "trainable": true, "dtype": "float32", "activation": "relu"}}
╟	
	╡axis

╢gamma
	╖beta
╕moving_mean
╣moving_variance
║regularization_losses
╗trainable_variables
╝	variables
╜	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"ш
_tf_keras_layer╬{"class_name": "BatchNormalization", "name": "batch_normalization_62", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_62", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 32]}}
╟	
	╛axis

┐gamma
	└beta
┴moving_mean
┬moving_variance
├regularization_losses
─trainable_variables
┼	variables
╞	keras_api
Я__call__
+а&call_and_return_all_conditional_losses"ш
_tf_keras_layer╬{"class_name": "BatchNormalization", "name": "batch_normalization_66", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_66", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 33, 33, 32]}}
З
╟regularization_losses
╚trainable_variables
╔	variables
╩	keras_api
б__call__
+в&call_and_return_all_conditional_losses"Є
_tf_keras_layer╪{"class_name": "MaxPooling2D", "name": "max_pooling2d_47", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_47", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
З
╦regularization_losses
╠trainable_variables
═	variables
╬	keras_api
г__call__
+д&call_and_return_all_conditional_losses"Є
_tf_keras_layer╪{"class_name": "MaxPooling2D", "name": "max_pooling2d_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_50", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ю
╧regularization_losses
╨trainable_variables
╤	variables
╥	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"class_name": "Dropout", "name": "dropout_62", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_62", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
ю
╙regularization_losses
╘trainable_variables
╒	variables
╓	keras_api
з__call__
+и&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"class_name": "Dropout", "name": "dropout_66", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_66", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
ю
╫regularization_losses
╪trainable_variables
┘	variables
┌	keras_api
й__call__
+к&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"class_name": "Flatten", "name": "flatten_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ю
█regularization_losses
▄trainable_variables
▌	variables
▐	keras_api
л__call__
+м&call_and_return_all_conditional_losses"┘
_tf_keras_layer┐{"class_name": "Flatten", "name": "flatten_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Б
▀kernel
	рbias
сregularization_losses
тtrainable_variables
у	variables
ф	keras_api
н__call__
+о&call_and_return_all_conditional_losses"╘
_tf_keras_layer║{"class_name": "Dense", "name": "dense_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8192]}}
Б
хkernel
	цbias
чregularization_losses
шtrainable_variables
щ	variables
ъ	keras_api
п__call__
+░&call_and_return_all_conditional_losses"╘
_tf_keras_layer║{"class_name": "Dense", "name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8192]}}
▌
ыregularization_losses
ьtrainable_variables
э	variables
ю	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses"╚
_tf_keras_layerо{"class_name": "Activation", "name": "activation_63", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_63", "trainable": true, "dtype": "float32", "activation": "relu"}}
▌
яregularization_losses
Ёtrainable_variables
ё	variables
Є	keras_api
│__call__
+┤&call_and_return_all_conditional_losses"╚
_tf_keras_layerо{"class_name": "Activation", "name": "activation_67", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_67", "trainable": true, "dtype": "float32", "activation": "relu"}}
┴	
	єaxis

Їgamma
	їbeta
Ўmoving_mean
ўmoving_variance
°regularization_losses
∙trainable_variables
·	variables
√	keras_api
╡__call__
+╢&call_and_return_all_conditional_losses"т
_tf_keras_layer╚{"class_name": "BatchNormalization", "name": "batch_normalization_63", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_63", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
┴	
	№axis

¤gamma
	■beta
 moving_mean
Аmoving_variance
Бregularization_losses
Вtrainable_variables
Г	variables
Д	keras_api
╖__call__
+╕&call_and_return_all_conditional_losses"т
_tf_keras_layer╚{"class_name": "BatchNormalization", "name": "batch_normalization_67", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_67", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
э
Еregularization_losses
Жtrainable_variables
З	variables
И	keras_api
╣__call__
+║&call_and_return_all_conditional_losses"╪
_tf_keras_layer╛{"class_name": "Dropout", "name": "dropout_63", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_63", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
э
Йregularization_losses
Кtrainable_variables
Л	variables
М	keras_api
╗__call__
+╝&call_and_return_all_conditional_losses"╪
_tf_keras_layer╛{"class_name": "Dropout", "name": "dropout_67", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_67", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
¤
Нkernel
	Оbias
Пregularization_losses
Рtrainable_variables
С	variables
Т	keras_api
╜__call__
+╛&call_and_return_all_conditional_losses"╨
_tf_keras_layer╢{"class_name": "Dense", "name": "dense_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
¤
Уkernel
	Фbias
Хregularization_losses
Цtrainable_variables
Ч	variables
Ш	keras_api
┐__call__
+└&call_and_return_all_conditional_losses"╨
_tf_keras_layer╢{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
┘
Щregularization_losses
Ъtrainable_variables
Ы	variables
Ь	keras_api
┴__call__
+┬&call_and_return_all_conditional_losses"─
_tf_keras_layerк{"class_name": "Activation", "name": "age_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "age_output", "trainable": true, "dtype": "float32", "activation": "linear"}}
р
Эregularization_losses
Юtrainable_variables
Я	variables
а	keras_api
├__call__
+─&call_and_return_all_conditional_losses"╦
_tf_keras_layer▒{"class_name": "Activation", "name": "gender_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gender_output", "trainable": true, "dtype": "float32", "activation": "sigmoid"}}
╪
	бiter
вbeta_1
гbeta_2

дdecay
еlearning_rate5mв6mг;mд<mеJmжKmзSmиTmйkmкlmлqmмrmн	Аmо	Бmп	Йm░	Кm▒	бm▓	вm│	зm┤	иm╡	╢m╢	╖m╖	┐m╕	└m╣	▀m║	рm╗	хm╝	цm╜	Їm╛	їm┐	¤m└	■m┴	Нm┬	Оm├	Уm─	Фm┼5v╞6v╟;v╚<v╔Jv╩Kv╦Sv╠Tv═kv╬lv╧qv╨rv╤	Аv╥	Бv╙	Йv╘	Кv╒	бv╓	вv╫	зv╪	иv┘	╢v┌	╖v█	┐v▄	└v▌	▀v▐	рv▀	хvр	цvс	Їvт	їvу	¤vф	■vх	Нvц	Оvч	Уvш	Фvщ"
	optimizer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
╬
50
61
;2
<3
J4
K5
S6
T7
k8
l9
q10
r11
А12
Б13
Й14
К15
б16
в17
з18
и19
╢20
╖21
┐22
└23
▀24
р25
х26
ц27
Ї28
ї29
¤30
■31
Н32
О33
У34
Ф35"
trackable_list_wrapper
┌
50
61
;2
<3
J4
K5
L6
M7
S8
T9
U10
V11
k12
l13
q14
r15
А16
Б17
В18
Г19
Й20
К21
Л22
М23
б24
в25
з26
и27
╢28
╖29
╕30
╣31
┐32
└33
┴34
┬35
▀36
р37
х38
ц39
Ї40
ї41
Ў42
ў43
¤44
■45
 46
А47
Н48
О49
У50
Ф51"
trackable_list_wrapper
╙
жlayer_metrics
0regularization_losses
 зlayer_regularization_losses
1trainable_variables
иmetrics
2	variables
йlayers
кnon_trainable_variables
ъ__call__
ь_default_save_signature
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
-
┼serving_default"
signature_map
*:(2conv2d_45/kernel
:2conv2d_45/bias
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
╡
лlayers
мlayer_metrics
 нlayer_regularization_losses
7regularization_losses
8trainable_variables
9	variables
оmetrics
пnon_trainable_variables
э__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_48/kernel
:2conv2d_48/bias
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
╡
░layers
▒layer_metrics
 ▓layer_regularization_losses
=regularization_losses
>trainable_variables
?	variables
│metrics
┤non_trainable_variables
я__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╡layers
╢layer_metrics
 ╖layer_regularization_losses
Aregularization_losses
Btrainable_variables
C	variables
╕metrics
╣non_trainable_variables
ё__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
║layers
╗layer_metrics
 ╝layer_regularization_losses
Eregularization_losses
Ftrainable_variables
G	variables
╜metrics
╛non_trainable_variables
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_60/gamma
):'2batch_normalization_60/beta
2:0 (2"batch_normalization_60/moving_mean
6:4 (2&batch_normalization_60/moving_variance
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
<
J0
K1
L2
M3"
trackable_list_wrapper
╡
┐layers
└layer_metrics
 ┴layer_regularization_losses
Nregularization_losses
Otrainable_variables
P	variables
┬metrics
├non_trainable_variables
ї__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_64/gamma
):'2batch_normalization_64/beta
2:0 (2"batch_normalization_64/moving_mean
6:4 (2&batch_normalization_64/moving_variance
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
<
S0
T1
U2
V3"
trackable_list_wrapper
╡
─layers
┼layer_metrics
 ╞layer_regularization_losses
Wregularization_losses
Xtrainable_variables
Y	variables
╟metrics
╚non_trainable_variables
ў__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╔layers
╩layer_metrics
 ╦layer_regularization_losses
[regularization_losses
\trainable_variables
]	variables
╠metrics
═non_trainable_variables
∙__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╬layers
╧layer_metrics
 ╨layer_regularization_losses
_regularization_losses
`trainable_variables
a	variables
╤metrics
╥non_trainable_variables
√__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╙layers
╘layer_metrics
 ╒layer_regularization_losses
cregularization_losses
dtrainable_variables
e	variables
╓metrics
╫non_trainable_variables
¤__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╪layers
┘layer_metrics
 ┌layer_regularization_losses
gregularization_losses
htrainable_variables
i	variables
█metrics
▄non_trainable_variables
 __call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_46/kernel
: 2conv2d_46/bias
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
╡
▌layers
▐layer_metrics
 ▀layer_regularization_losses
mregularization_losses
ntrainable_variables
o	variables
рmetrics
сnon_trainable_variables
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_49/kernel
: 2conv2d_49/bias
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
╡
тlayers
уlayer_metrics
 фlayer_regularization_losses
sregularization_losses
ttrainable_variables
u	variables
хmetrics
цnon_trainable_variables
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
чlayers
шlayer_metrics
 щlayer_regularization_losses
wregularization_losses
xtrainable_variables
y	variables
ъmetrics
ыnon_trainable_variables
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
ьlayers
эlayer_metrics
 юlayer_regularization_losses
{regularization_losses
|trainable_variables
}	variables
яmetrics
Ёnon_trainable_variables
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_61/gamma
):' 2batch_normalization_61/beta
2:0  (2"batch_normalization_61/moving_mean
6:4  (2&batch_normalization_61/moving_variance
 "
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
@
А0
Б1
В2
Г3"
trackable_list_wrapper
╕
ёlayers
Єlayer_metrics
 єlayer_regularization_losses
Дregularization_losses
Еtrainable_variables
Ж	variables
Їmetrics
їnon_trainable_variables
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_65/gamma
):' 2batch_normalization_65/beta
2:0  (2"batch_normalization_65/moving_mean
6:4  (2&batch_normalization_65/moving_variance
 "
trackable_list_wrapper
0
Й0
К1"
trackable_list_wrapper
@
Й0
К1
Л2
М3"
trackable_list_wrapper
╕
Ўlayers
ўlayer_metrics
 °layer_regularization_losses
Нregularization_losses
Оtrainable_variables
П	variables
∙metrics
·non_trainable_variables
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
√layers
№layer_metrics
 ¤layer_regularization_losses
Сregularization_losses
Тtrainable_variables
У	variables
■metrics
 non_trainable_variables
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Аlayers
Бlayer_metrics
 Вlayer_regularization_losses
Хregularization_losses
Цtrainable_variables
Ч	variables
Гmetrics
Дnon_trainable_variables
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Еlayers
Жlayer_metrics
 Зlayer_regularization_losses
Щregularization_losses
Ъtrainable_variables
Ы	variables
Иmetrics
Йnon_trainable_variables
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Кlayers
Лlayer_metrics
 Мlayer_regularization_losses
Эregularization_losses
Юtrainable_variables
Я	variables
Нmetrics
Оnon_trainable_variables
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_47/kernel
: 2conv2d_47/bias
 "
trackable_list_wrapper
0
б0
в1"
trackable_list_wrapper
0
б0
в1"
trackable_list_wrapper
╕
Пlayers
Рlayer_metrics
 Сlayer_regularization_losses
гregularization_losses
дtrainable_variables
е	variables
Тmetrics
Уnon_trainable_variables
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_50/kernel
: 2conv2d_50/bias
 "
trackable_list_wrapper
0
з0
и1"
trackable_list_wrapper
0
з0
и1"
trackable_list_wrapper
╕
Фlayers
Хlayer_metrics
 Цlayer_regularization_losses
йregularization_losses
кtrainable_variables
л	variables
Чmetrics
Шnon_trainable_variables
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Щlayers
Ъlayer_metrics
 Ыlayer_regularization_losses
нregularization_losses
оtrainable_variables
п	variables
Ьmetrics
Эnon_trainable_variables
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Юlayers
Яlayer_metrics
 аlayer_regularization_losses
▒regularization_losses
▓trainable_variables
│	variables
бmetrics
вnon_trainable_variables
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_62/gamma
):' 2batch_normalization_62/beta
2:0  (2"batch_normalization_62/moving_mean
6:4  (2&batch_normalization_62/moving_variance
 "
trackable_list_wrapper
0
╢0
╖1"
trackable_list_wrapper
@
╢0
╖1
╕2
╣3"
trackable_list_wrapper
╕
гlayers
дlayer_metrics
 еlayer_regularization_losses
║regularization_losses
╗trainable_variables
╝	variables
жmetrics
зnon_trainable_variables
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_66/gamma
):' 2batch_normalization_66/beta
2:0  (2"batch_normalization_66/moving_mean
6:4  (2&batch_normalization_66/moving_variance
 "
trackable_list_wrapper
0
┐0
└1"
trackable_list_wrapper
@
┐0
└1
┴2
┬3"
trackable_list_wrapper
╕
иlayers
йlayer_metrics
 кlayer_regularization_losses
├regularization_losses
─trainable_variables
┼	variables
лmetrics
мnon_trainable_variables
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
нlayers
оlayer_metrics
 пlayer_regularization_losses
╟regularization_losses
╚trainable_variables
╔	variables
░metrics
▒non_trainable_variables
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▓layers
│layer_metrics
 ┤layer_regularization_losses
╦regularization_losses
╠trainable_variables
═	variables
╡metrics
╢non_trainable_variables
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╖layers
╕layer_metrics
 ╣layer_regularization_losses
╧regularization_losses
╨trainable_variables
╤	variables
║metrics
╗non_trainable_variables
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╝layers
╜layer_metrics
 ╛layer_regularization_losses
╙regularization_losses
╘trainable_variables
╒	variables
┐metrics
└non_trainable_variables
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┴layers
┬layer_metrics
 ├layer_regularization_losses
╫regularization_losses
╪trainable_variables
┘	variables
─metrics
┼non_trainable_variables
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╞layers
╟layer_metrics
 ╚layer_regularization_losses
█regularization_losses
▄trainable_variables
▌	variables
╔metrics
╩non_trainable_variables
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
#:!
А@А2dense_30/kernel
:А2dense_30/bias
 "
trackable_list_wrapper
0
▀0
р1"
trackable_list_wrapper
0
▀0
р1"
trackable_list_wrapper
╕
╦layers
╠layer_metrics
 ═layer_regularization_losses
сregularization_losses
тtrainable_variables
у	variables
╬metrics
╧non_trainable_variables
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
#:!
А@А2dense_32/kernel
:А2dense_32/bias
 "
trackable_list_wrapper
0
х0
ц1"
trackable_list_wrapper
0
х0
ц1"
trackable_list_wrapper
╕
╨layers
╤layer_metrics
 ╥layer_regularization_losses
чregularization_losses
шtrainable_variables
щ	variables
╙metrics
╘non_trainable_variables
п__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╒layers
╓layer_metrics
 ╫layer_regularization_losses
ыregularization_losses
ьtrainable_variables
э	variables
╪metrics
┘non_trainable_variables
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┌layers
█layer_metrics
 ▄layer_regularization_losses
яregularization_losses
Ёtrainable_variables
ё	variables
▌metrics
▐non_trainable_variables
│__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)А2batch_normalization_63/gamma
*:(А2batch_normalization_63/beta
3:1А (2"batch_normalization_63/moving_mean
7:5А (2&batch_normalization_63/moving_variance
 "
trackable_list_wrapper
0
Ї0
ї1"
trackable_list_wrapper
@
Ї0
ї1
Ў2
ў3"
trackable_list_wrapper
╕
▀layers
рlayer_metrics
 сlayer_regularization_losses
°regularization_losses
∙trainable_variables
·	variables
тmetrics
уnon_trainable_variables
╡__call__
+╢&call_and_return_all_conditional_losses
'╢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)А2batch_normalization_67/gamma
*:(А2batch_normalization_67/beta
3:1А (2"batch_normalization_67/moving_mean
7:5А (2&batch_normalization_67/moving_variance
 "
trackable_list_wrapper
0
¤0
■1"
trackable_list_wrapper
@
¤0
■1
 2
А3"
trackable_list_wrapper
╕
фlayers
хlayer_metrics
 цlayer_regularization_losses
Бregularization_losses
Вtrainable_variables
Г	variables
чmetrics
шnon_trainable_variables
╖__call__
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
щlayers
ъlayer_metrics
 ыlayer_regularization_losses
Еregularization_losses
Жtrainable_variables
З	variables
ьmetrics
эnon_trainable_variables
╣__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
юlayers
яlayer_metrics
 Ёlayer_regularization_losses
Йregularization_losses
Кtrainable_variables
Л	variables
ёmetrics
Єnon_trainable_variables
╗__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
": 	А2dense_31/kernel
:2dense_31/bias
 "
trackable_list_wrapper
0
Н0
О1"
trackable_list_wrapper
0
Н0
О1"
trackable_list_wrapper
╕
єlayers
Їlayer_metrics
 їlayer_regularization_losses
Пregularization_losses
Рtrainable_variables
С	variables
Ўmetrics
ўnon_trainable_variables
╜__call__
+╛&call_and_return_all_conditional_losses
'╛"call_and_return_conditional_losses"
_generic_user_object
": 	А2dense_33/kernel
:2dense_33/bias
 "
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
╕
°layers
∙layer_metrics
 ·layer_regularization_losses
Хregularization_losses
Цtrainable_variables
Ч	variables
√metrics
№non_trainable_variables
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
¤layers
■layer_metrics
  layer_regularization_losses
Щregularization_losses
Ъtrainable_variables
Ы	variables
Аmetrics
Бnon_trainable_variables
┴__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Вlayers
Гlayer_metrics
 Дlayer_regularization_losses
Эregularization_losses
Юtrainable_variables
Я	variables
Еmetrics
Жnon_trainable_variables
├__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
H
З0
И1
Й2
К3
Л4"
trackable_list_wrapper
■
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44"
trackable_list_wrapper
в
L0
M1
U2
V3
В4
Г5
Л6
М7
╕8
╣9
┴10
┬11
Ў12
ў13
 14
А15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Л0
М1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
╕0
╣1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
┴0
┬1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ў0
ў1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
 0
А1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┐

Мtotal

Нcount
О	variables
П	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
╓

Рtotal

Сcount
Т	variables
У	keras_api"Ы
_tf_keras_metricА{"class_name": "Mean", "name": "age_output_loss", "dtype": "float32", "config": {"name": "age_output_loss", "dtype": "float32"}}
▄

Фtotal

Хcount
Ц	variables
Ч	keras_api"б
_tf_keras_metricЖ{"class_name": "Mean", "name": "gender_output_loss", "dtype": "float32", "config": {"name": "gender_output_loss", "dtype": "float32"}}
П

Шtotal

Щcount
Ъ
_fn_kwargs
Ы	variables
Ь	keras_api"├
_tf_keras_metricи{"class_name": "MeanMetricWrapper", "name": "age_output_mae", "dtype": "float32", "config": {"name": "age_output_mae", "dtype": "float32", "fn": "mean_absolute_error"}}
а

Эtotal

Юcount
Я
_fn_kwargs
а	variables
б	keras_api"╘
_tf_keras_metric╣{"class_name": "MeanMetricWrapper", "name": "gender_output_accuracy", "dtype": "float32", "config": {"name": "gender_output_accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
М0
Н1"
trackable_list_wrapper
.
О	variables"
_generic_user_object
:  (2total
:  (2count
0
Р0
С1"
trackable_list_wrapper
.
Т	variables"
_generic_user_object
:  (2total
:  (2count
0
Ф0
Х1"
trackable_list_wrapper
.
Ц	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ш0
Щ1"
trackable_list_wrapper
.
Ы	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Э0
Ю1"
trackable_list_wrapper
.
а	variables"
_generic_user_object
/:-2Adam/conv2d_45/kernel/m
!:2Adam/conv2d_45/bias/m
/:-2Adam/conv2d_48/kernel/m
!:2Adam/conv2d_48/bias/m
/:-2#Adam/batch_normalization_60/gamma/m
.:,2"Adam/batch_normalization_60/beta/m
/:-2#Adam/batch_normalization_64/gamma/m
.:,2"Adam/batch_normalization_64/beta/m
/:- 2Adam/conv2d_46/kernel/m
!: 2Adam/conv2d_46/bias/m
/:- 2Adam/conv2d_49/kernel/m
!: 2Adam/conv2d_49/bias/m
/:- 2#Adam/batch_normalization_61/gamma/m
.:, 2"Adam/batch_normalization_61/beta/m
/:- 2#Adam/batch_normalization_65/gamma/m
.:, 2"Adam/batch_normalization_65/beta/m
/:-  2Adam/conv2d_47/kernel/m
!: 2Adam/conv2d_47/bias/m
/:-  2Adam/conv2d_50/kernel/m
!: 2Adam/conv2d_50/bias/m
/:- 2#Adam/batch_normalization_62/gamma/m
.:, 2"Adam/batch_normalization_62/beta/m
/:- 2#Adam/batch_normalization_66/gamma/m
.:, 2"Adam/batch_normalization_66/beta/m
(:&
А@А2Adam/dense_30/kernel/m
!:А2Adam/dense_30/bias/m
(:&
А@А2Adam/dense_32/kernel/m
!:А2Adam/dense_32/bias/m
0:.А2#Adam/batch_normalization_63/gamma/m
/:-А2"Adam/batch_normalization_63/beta/m
0:.А2#Adam/batch_normalization_67/gamma/m
/:-А2"Adam/batch_normalization_67/beta/m
':%	А2Adam/dense_31/kernel/m
 :2Adam/dense_31/bias/m
':%	А2Adam/dense_33/kernel/m
 :2Adam/dense_33/bias/m
/:-2Adam/conv2d_45/kernel/v
!:2Adam/conv2d_45/bias/v
/:-2Adam/conv2d_48/kernel/v
!:2Adam/conv2d_48/bias/v
/:-2#Adam/batch_normalization_60/gamma/v
.:,2"Adam/batch_normalization_60/beta/v
/:-2#Adam/batch_normalization_64/gamma/v
.:,2"Adam/batch_normalization_64/beta/v
/:- 2Adam/conv2d_46/kernel/v
!: 2Adam/conv2d_46/bias/v
/:- 2Adam/conv2d_49/kernel/v
!: 2Adam/conv2d_49/bias/v
/:- 2#Adam/batch_normalization_61/gamma/v
.:, 2"Adam/batch_normalization_61/beta/v
/:- 2#Adam/batch_normalization_65/gamma/v
.:, 2"Adam/batch_normalization_65/beta/v
/:-  2Adam/conv2d_47/kernel/v
!: 2Adam/conv2d_47/bias/v
/:-  2Adam/conv2d_50/kernel/v
!: 2Adam/conv2d_50/bias/v
/:- 2#Adam/batch_normalization_62/gamma/v
.:, 2"Adam/batch_normalization_62/beta/v
/:- 2#Adam/batch_normalization_66/gamma/v
.:, 2"Adam/batch_normalization_66/beta/v
(:&
А@А2Adam/dense_30/kernel/v
!:А2Adam/dense_30/bias/v
(:&
А@А2Adam/dense_32/kernel/v
!:А2Adam/dense_32/bias/v
0:.А2#Adam/batch_normalization_63/gamma/v
/:-А2"Adam/batch_normalization_63/beta/v
0:.А2#Adam/batch_normalization_67/gamma/v
/:-А2"Adam/batch_normalization_67/beta/v
':%	А2Adam/dense_31/kernel/v
 :2Adam/dense_31/bias/v
':%	А2Adam/dense_33/kernel/v
 :2Adam/dense_33/bias/v
ю2ы
(__inference_face_net_layer_call_fn_41421
(__inference_face_net_layer_call_fn_40409
(__inference_face_net_layer_call_fn_40673
(__inference_face_net_layer_call_fn_41532└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┌2╫
C__inference_face_net_layer_call_and_return_conditional_losses_41102
C__inference_face_net_layer_call_and_return_conditional_losses_41310
C__inference_face_net_layer_call_and_return_conditional_losses_39991
C__inference_face_net_layer_call_and_return_conditional_losses_40144└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ш2х
 __inference__wrapped_model_37838└
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *0в-
+К(
input_6         ╞╞
╙2╨
)__inference_conv2d_45_layer_call_fn_41551в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_conv2d_45_layer_call_and_return_conditional_losses_41542в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_conv2d_48_layer_call_fn_41570в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_conv2d_48_layer_call_and_return_conditional_losses_41561в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘
-__inference_activation_60_layer_call_fn_41580в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Є2я
H__inference_activation_60_layer_call_and_return_conditional_losses_41575в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘
-__inference_activation_64_layer_call_fn_41590в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Є2я
H__inference_activation_64_layer_call_and_return_conditional_losses_41585в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ъ2Ч
6__inference_batch_normalization_60_layer_call_fn_41718
6__inference_batch_normalization_60_layer_call_fn_41654
6__inference_batch_normalization_60_layer_call_fn_41705
6__inference_batch_normalization_60_layer_call_fn_41641┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ж2Г
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41610
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41628
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41692
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41674┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ъ2Ч
6__inference_batch_normalization_64_layer_call_fn_41833
6__inference_batch_normalization_64_layer_call_fn_41782
6__inference_batch_normalization_64_layer_call_fn_41769
6__inference_batch_normalization_64_layer_call_fn_41846┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ж2Г
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41756
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41820
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41738
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41802┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ш2Х
0__inference_max_pooling2d_45_layer_call_fn_38058р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
│2░
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_38052р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_48_layer_call_fn_38070р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
│2░
K__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_38064р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Т2П
*__inference_dropout_60_layer_call_fn_41868
*__inference_dropout_60_layer_call_fn_41873┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╚2┼
E__inference_dropout_60_layer_call_and_return_conditional_losses_41858
E__inference_dropout_60_layer_call_and_return_conditional_losses_41863┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
*__inference_dropout_64_layer_call_fn_41900
*__inference_dropout_64_layer_call_fn_41895┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╚2┼
E__inference_dropout_64_layer_call_and_return_conditional_losses_41890
E__inference_dropout_64_layer_call_and_return_conditional_losses_41885┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╙2╨
)__inference_conv2d_46_layer_call_fn_41919в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_conv2d_46_layer_call_and_return_conditional_losses_41910в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_conv2d_49_layer_call_fn_41938в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_conv2d_49_layer_call_and_return_conditional_losses_41929в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘
-__inference_activation_61_layer_call_fn_41948в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Є2я
H__inference_activation_61_layer_call_and_return_conditional_losses_41943в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘
-__inference_activation_65_layer_call_fn_41958в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Є2я
H__inference_activation_65_layer_call_and_return_conditional_losses_41953в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ъ2Ч
6__inference_batch_normalization_61_layer_call_fn_42022
6__inference_batch_normalization_61_layer_call_fn_42073
6__inference_batch_normalization_61_layer_call_fn_42086
6__inference_batch_normalization_61_layer_call_fn_42009┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ж2Г
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41978
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41996
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_42060
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_42042┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ъ2Ч
6__inference_batch_normalization_65_layer_call_fn_42137
6__inference_batch_normalization_65_layer_call_fn_42201
6__inference_batch_normalization_65_layer_call_fn_42150
6__inference_batch_normalization_65_layer_call_fn_42214┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ж2Г
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42106
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42124
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42170
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42188┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ш2Х
0__inference_max_pooling2d_46_layer_call_fn_38290р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
│2░
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_38284р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_49_layer_call_fn_38302р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
│2░
K__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_38296р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Т2П
*__inference_dropout_61_layer_call_fn_42241
*__inference_dropout_61_layer_call_fn_42236┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╚2┼
E__inference_dropout_61_layer_call_and_return_conditional_losses_42226
E__inference_dropout_61_layer_call_and_return_conditional_losses_42231┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
*__inference_dropout_65_layer_call_fn_42263
*__inference_dropout_65_layer_call_fn_42268┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╚2┼
E__inference_dropout_65_layer_call_and_return_conditional_losses_42258
E__inference_dropout_65_layer_call_and_return_conditional_losses_42253┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╙2╨
)__inference_conv2d_47_layer_call_fn_42287в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_conv2d_47_layer_call_and_return_conditional_losses_42278в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_conv2d_50_layer_call_fn_42306в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_conv2d_50_layer_call_and_return_conditional_losses_42297в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘
-__inference_activation_62_layer_call_fn_42316в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Є2я
H__inference_activation_62_layer_call_and_return_conditional_losses_42311в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘
-__inference_activation_66_layer_call_fn_42326в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Є2я
H__inference_activation_66_layer_call_and_return_conditional_losses_42321в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ъ2Ч
6__inference_batch_normalization_62_layer_call_fn_42390
6__inference_batch_normalization_62_layer_call_fn_42441
6__inference_batch_normalization_62_layer_call_fn_42377
6__inference_batch_normalization_62_layer_call_fn_42454┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ж2Г
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42346
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42428
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42364
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42410┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ъ2Ч
6__inference_batch_normalization_66_layer_call_fn_42505
6__inference_batch_normalization_66_layer_call_fn_42569
6__inference_batch_normalization_66_layer_call_fn_42582
6__inference_batch_normalization_66_layer_call_fn_42518┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ж2Г
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42538
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42474
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42492
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42556┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ш2Х
0__inference_max_pooling2d_47_layer_call_fn_38522р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
│2░
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_38516р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Ш2Х
0__inference_max_pooling2d_50_layer_call_fn_38534р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
│2░
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_38528р
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *@в=
;К84                                    
Т2П
*__inference_dropout_62_layer_call_fn_42604
*__inference_dropout_62_layer_call_fn_42609┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╚2┼
E__inference_dropout_62_layer_call_and_return_conditional_losses_42599
E__inference_dropout_62_layer_call_and_return_conditional_losses_42594┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
*__inference_dropout_66_layer_call_fn_42636
*__inference_dropout_66_layer_call_fn_42631┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╚2┼
E__inference_dropout_66_layer_call_and_return_conditional_losses_42621
E__inference_dropout_66_layer_call_and_return_conditional_losses_42626┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
*__inference_flatten_15_layer_call_fn_42647в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_flatten_15_layer_call_and_return_conditional_losses_42642в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_flatten_16_layer_call_fn_42658в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_flatten_16_layer_call_and_return_conditional_losses_42653в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_30_layer_call_fn_42677в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_30_layer_call_and_return_conditional_losses_42668в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_32_layer_call_fn_42696в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_32_layer_call_and_return_conditional_losses_42687в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘
-__inference_activation_63_layer_call_fn_42706в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Є2я
H__inference_activation_63_layer_call_and_return_conditional_losses_42701в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘
-__inference_activation_67_layer_call_fn_42716в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Є2я
H__inference_activation_67_layer_call_and_return_conditional_losses_42711в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
к2з
6__inference_batch_normalization_63_layer_call_fn_42785
6__inference_batch_normalization_63_layer_call_fn_42798┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
р2▌
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_42752
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_42772┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
к2з
6__inference_batch_normalization_67_layer_call_fn_42867
6__inference_batch_normalization_67_layer_call_fn_42880┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
р2▌
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_42854
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_42834┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
*__inference_dropout_63_layer_call_fn_42907
*__inference_dropout_63_layer_call_fn_42902┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╚2┼
E__inference_dropout_63_layer_call_and_return_conditional_losses_42892
E__inference_dropout_63_layer_call_and_return_conditional_losses_42897┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Т2П
*__inference_dropout_67_layer_call_fn_42934
*__inference_dropout_67_layer_call_fn_42929┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╚2┼
E__inference_dropout_67_layer_call_and_return_conditional_losses_42919
E__inference_dropout_67_layer_call_and_return_conditional_losses_42924┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_dense_31_layer_call_fn_42953в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_31_layer_call_and_return_conditional_losses_42944в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_33_layer_call_fn_42972в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_33_layer_call_and_return_conditional_losses_42963в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_age_output_layer_call_fn_42981в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_age_output_layer_call_and_return_conditional_losses_42976в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘
-__inference_gender_output_layer_call_fn_42991в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Є2я
H__inference_gender_output_layer_call_and_return_conditional_losses_42986в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╩B╟
#__inference_signature_wrapper_40794input_6"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 о
 __inference__wrapped_model_37838ЙX;<56STUVJKLMqrklЙКЛМАБВГзибв┐└┴┬╢╖╕╣хц▀рА¤ ■ўЇЎїУФНО:в7
0в-
+К(
input_6         ╞╞
к "qкn
2

age_output$К!

age_output         
8
gender_output'К$
gender_output         ╕
H__inference_activation_60_layer_call_and_return_conditional_losses_41575l9в6
/в,
*К'
inputs         ╞╞
к "/в,
%К"
0         ╞╞
Ъ Р
-__inference_activation_60_layer_call_fn_41580_9в6
/в,
*К'
inputs         ╞╞
к ""К         ╞╞┤
H__inference_activation_61_layer_call_and_return_conditional_losses_41943h7в4
-в*
(К%
inputs         BB 
к "-в*
#К 
0         BB 
Ъ М
-__inference_activation_61_layer_call_fn_41948[7в4
-в*
(К%
inputs         BB 
к " К         BB ┤
H__inference_activation_62_layer_call_and_return_conditional_losses_42311h7в4
-в*
(К%
inputs         !! 
к "-в*
#К 
0         !! 
Ъ М
-__inference_activation_62_layer_call_fn_42316[7в4
-в*
(К%
inputs         !! 
к " К         !! ж
H__inference_activation_63_layer_call_and_return_conditional_losses_42701Z0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ ~
-__inference_activation_63_layer_call_fn_42706M0в-
&в#
!К
inputs         А
к "К         А╕
H__inference_activation_64_layer_call_and_return_conditional_losses_41585l9в6
/в,
*К'
inputs         ╞╞
к "/в,
%К"
0         ╞╞
Ъ Р
-__inference_activation_64_layer_call_fn_41590_9в6
/в,
*К'
inputs         ╞╞
к ""К         ╞╞┤
H__inference_activation_65_layer_call_and_return_conditional_losses_41953h7в4
-в*
(К%
inputs         BB 
к "-в*
#К 
0         BB 
Ъ М
-__inference_activation_65_layer_call_fn_41958[7в4
-в*
(К%
inputs         BB 
к " К         BB ┤
H__inference_activation_66_layer_call_and_return_conditional_losses_42321h7в4
-в*
(К%
inputs         !! 
к "-в*
#К 
0         !! 
Ъ М
-__inference_activation_66_layer_call_fn_42326[7в4
-в*
(К%
inputs         !! 
к " К         !! ж
H__inference_activation_67_layer_call_and_return_conditional_losses_42711Z0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ ~
-__inference_activation_67_layer_call_fn_42716M0в-
&в#
!К
inputs         А
к "К         Аб
E__inference_age_output_layer_call_and_return_conditional_losses_42976X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ y
*__inference_age_output_layer_call_fn_42981K/в,
%в"
 К
inputs         
к "К         ╦
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41610vJKLM=в:
3в0
*К'
inputs         ╞╞
p
к "/в,
%К"
0         ╞╞
Ъ ╦
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41628vJKLM=в:
3в0
*К'
inputs         ╞╞
p 
к "/в,
%К"
0         ╞╞
Ъ ь
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41674ЦJKLMMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ь
Q__inference_batch_normalization_60_layer_call_and_return_conditional_losses_41692ЦJKLMMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ г
6__inference_batch_normalization_60_layer_call_fn_41641iJKLM=в:
3в0
*К'
inputs         ╞╞
p
к ""К         ╞╞г
6__inference_batch_normalization_60_layer_call_fn_41654iJKLM=в:
3в0
*К'
inputs         ╞╞
p 
к ""К         ╞╞─
6__inference_batch_normalization_60_layer_call_fn_41705ЙJKLMMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ─
6__inference_batch_normalization_60_layer_call_fn_41718ЙJKLMMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           Ё
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41978ЪАБВГMвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ Ё
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_41996ЪАБВГMвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ ╦
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_42042vАБВГ;в8
1в.
(К%
inputs         BB 
p
к "-в*
#К 
0         BB 
Ъ ╦
Q__inference_batch_normalization_61_layer_call_and_return_conditional_losses_42060vАБВГ;в8
1в.
(К%
inputs         BB 
p 
к "-в*
#К 
0         BB 
Ъ ╚
6__inference_batch_normalization_61_layer_call_fn_42009НАБВГMвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ╚
6__inference_batch_normalization_61_layer_call_fn_42022НАБВГMвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            г
6__inference_batch_normalization_61_layer_call_fn_42073iАБВГ;в8
1в.
(К%
inputs         BB 
p
к " К         BB г
6__inference_batch_normalization_61_layer_call_fn_42086iАБВГ;в8
1в.
(К%
inputs         BB 
p 
к " К         BB ╦
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42346v╢╖╕╣;в8
1в.
(К%
inputs         !! 
p
к "-в*
#К 
0         !! 
Ъ ╦
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42364v╢╖╕╣;в8
1в.
(К%
inputs         !! 
p 
к "-в*
#К 
0         !! 
Ъ Ё
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42410Ъ╢╖╕╣MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ Ё
Q__inference_batch_normalization_62_layer_call_and_return_conditional_losses_42428Ъ╢╖╕╣MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ г
6__inference_batch_normalization_62_layer_call_fn_42377i╢╖╕╣;в8
1в.
(К%
inputs         !! 
p
к " К         !! г
6__inference_batch_normalization_62_layer_call_fn_42390i╢╖╕╣;в8
1в.
(К%
inputs         !! 
p 
к " К         !! ╚
6__inference_batch_normalization_62_layer_call_fn_42441Н╢╖╕╣MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ╚
6__inference_batch_normalization_62_layer_call_fn_42454Н╢╖╕╣MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            ╜
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_42752hЎўЇї4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ ╜
Q__inference_batch_normalization_63_layer_call_and_return_conditional_losses_42772hўЇЎї4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ Х
6__inference_batch_normalization_63_layer_call_fn_42785[ЎўЇї4в1
*в'
!К
inputs         А
p
к "К         АХ
6__inference_batch_normalization_63_layer_call_fn_42798[ўЇЎї4в1
*в'
!К
inputs         А
p 
к "К         А╦
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41738vSTUV=в:
3в0
*К'
inputs         ╞╞
p
к "/в,
%К"
0         ╞╞
Ъ ╦
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41756vSTUV=в:
3в0
*К'
inputs         ╞╞
p 
к "/в,
%К"
0         ╞╞
Ъ ь
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41802ЦSTUVMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ь
Q__inference_batch_normalization_64_layer_call_and_return_conditional_losses_41820ЦSTUVMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ г
6__inference_batch_normalization_64_layer_call_fn_41769iSTUV=в:
3в0
*К'
inputs         ╞╞
p
к ""К         ╞╞г
6__inference_batch_normalization_64_layer_call_fn_41782iSTUV=в:
3в0
*К'
inputs         ╞╞
p 
к ""К         ╞╞─
6__inference_batch_normalization_64_layer_call_fn_41833ЙSTUVMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ─
6__inference_batch_normalization_64_layer_call_fn_41846ЙSTUVMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           Ё
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42106ЪЙКЛМMвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ Ё
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42124ЪЙКЛМMвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ ╦
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42170vЙКЛМ;в8
1в.
(К%
inputs         BB 
p
к "-в*
#К 
0         BB 
Ъ ╦
Q__inference_batch_normalization_65_layer_call_and_return_conditional_losses_42188vЙКЛМ;в8
1в.
(К%
inputs         BB 
p 
к "-в*
#К 
0         BB 
Ъ ╚
6__inference_batch_normalization_65_layer_call_fn_42137НЙКЛМMвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ╚
6__inference_batch_normalization_65_layer_call_fn_42150НЙКЛМMвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            г
6__inference_batch_normalization_65_layer_call_fn_42201iЙКЛМ;в8
1в.
(К%
inputs         BB 
p
к " К         BB г
6__inference_batch_normalization_65_layer_call_fn_42214iЙКЛМ;в8
1в.
(К%
inputs         BB 
p 
к " К         BB Ё
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42474Ъ┐└┴┬MвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ Ё
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42492Ъ┐└┴┬MвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ ╦
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42538v┐└┴┬;в8
1в.
(К%
inputs         !! 
p
к "-в*
#К 
0         !! 
Ъ ╦
Q__inference_batch_normalization_66_layer_call_and_return_conditional_losses_42556v┐└┴┬;в8
1в.
(К%
inputs         !! 
p 
к "-в*
#К 
0         !! 
Ъ ╚
6__inference_batch_normalization_66_layer_call_fn_42505Н┐└┴┬MвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ╚
6__inference_batch_normalization_66_layer_call_fn_42518Н┐└┴┬MвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            г
6__inference_batch_normalization_66_layer_call_fn_42569i┐└┴┬;в8
1в.
(К%
inputs         !! 
p
к " К         !! г
6__inference_batch_normalization_66_layer_call_fn_42582i┐└┴┬;в8
1в.
(К%
inputs         !! 
p 
к " К         !! ╜
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_42834h А¤■4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ ╜
Q__inference_batch_normalization_67_layer_call_and_return_conditional_losses_42854hА¤ ■4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ Х
6__inference_batch_normalization_67_layer_call_fn_42867[ А¤■4в1
*в'
!К
inputs         А
p
к "К         АХ
6__inference_batch_normalization_67_layer_call_fn_42880[А¤ ■4в1
*в'
!К
inputs         А
p 
к "К         А╕
D__inference_conv2d_45_layer_call_and_return_conditional_losses_41542p569в6
/в,
*К'
inputs         ╞╞
к "/в,
%К"
0         ╞╞
Ъ Р
)__inference_conv2d_45_layer_call_fn_41551c569в6
/в,
*К'
inputs         ╞╞
к ""К         ╞╞┤
D__inference_conv2d_46_layer_call_and_return_conditional_losses_41910lkl7в4
-в*
(К%
inputs         BB
к "-в*
#К 
0         BB 
Ъ М
)__inference_conv2d_46_layer_call_fn_41919_kl7в4
-в*
(К%
inputs         BB
к " К         BB ╢
D__inference_conv2d_47_layer_call_and_return_conditional_losses_42278nбв7в4
-в*
(К%
inputs         !! 
к "-в*
#К 
0         !! 
Ъ О
)__inference_conv2d_47_layer_call_fn_42287aбв7в4
-в*
(К%
inputs         !! 
к " К         !! ╕
D__inference_conv2d_48_layer_call_and_return_conditional_losses_41561p;<9в6
/в,
*К'
inputs         ╞╞
к "/в,
%К"
0         ╞╞
Ъ Р
)__inference_conv2d_48_layer_call_fn_41570c;<9в6
/в,
*К'
inputs         ╞╞
к ""К         ╞╞┤
D__inference_conv2d_49_layer_call_and_return_conditional_losses_41929lqr7в4
-в*
(К%
inputs         BB
к "-в*
#К 
0         BB 
Ъ М
)__inference_conv2d_49_layer_call_fn_41938_qr7в4
-в*
(К%
inputs         BB
к " К         BB ╢
D__inference_conv2d_50_layer_call_and_return_conditional_losses_42297nзи7в4
-в*
(К%
inputs         !! 
к "-в*
#К 
0         !! 
Ъ О
)__inference_conv2d_50_layer_call_fn_42306aзи7в4
-в*
(К%
inputs         !! 
к " К         !! з
C__inference_dense_30_layer_call_and_return_conditional_losses_42668`▀р0в-
&в#
!К
inputs         А@
к "&в#
К
0         А
Ъ 
(__inference_dense_30_layer_call_fn_42677S▀р0в-
&в#
!К
inputs         А@
к "К         Аж
C__inference_dense_31_layer_call_and_return_conditional_losses_42944_НО0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ ~
(__inference_dense_31_layer_call_fn_42953RНО0в-
&в#
!К
inputs         А
к "К         з
C__inference_dense_32_layer_call_and_return_conditional_losses_42687`хц0в-
&в#
!К
inputs         А@
к "&в#
К
0         А
Ъ 
(__inference_dense_32_layer_call_fn_42696Sхц0в-
&в#
!К
inputs         А@
к "К         Аж
C__inference_dense_33_layer_call_and_return_conditional_losses_42963_УФ0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ ~
(__inference_dense_33_layer_call_fn_42972RУФ0в-
&в#
!К
inputs         А
к "К         ╡
E__inference_dropout_60_layer_call_and_return_conditional_losses_41858l;в8
1в.
(К%
inputs         BB
p
к "-в*
#К 
0         BB
Ъ ╡
E__inference_dropout_60_layer_call_and_return_conditional_losses_41863l;в8
1в.
(К%
inputs         BB
p 
к "-в*
#К 
0         BB
Ъ Н
*__inference_dropout_60_layer_call_fn_41868_;в8
1в.
(К%
inputs         BB
p
к " К         BBН
*__inference_dropout_60_layer_call_fn_41873_;в8
1в.
(К%
inputs         BB
p 
к " К         BB╡
E__inference_dropout_61_layer_call_and_return_conditional_losses_42226l;в8
1в.
(К%
inputs         !! 
p
к "-в*
#К 
0         !! 
Ъ ╡
E__inference_dropout_61_layer_call_and_return_conditional_losses_42231l;в8
1в.
(К%
inputs         !! 
p 
к "-в*
#К 
0         !! 
Ъ Н
*__inference_dropout_61_layer_call_fn_42236_;в8
1в.
(К%
inputs         !! 
p
к " К         !! Н
*__inference_dropout_61_layer_call_fn_42241_;в8
1в.
(К%
inputs         !! 
p 
к " К         !! ╡
E__inference_dropout_62_layer_call_and_return_conditional_losses_42594l;в8
1в.
(К%
inputs          
p
к "-в*
#К 
0          
Ъ ╡
E__inference_dropout_62_layer_call_and_return_conditional_losses_42599l;в8
1в.
(К%
inputs          
p 
к "-в*
#К 
0          
Ъ Н
*__inference_dropout_62_layer_call_fn_42604_;в8
1в.
(К%
inputs          
p
к " К          Н
*__inference_dropout_62_layer_call_fn_42609_;в8
1в.
(К%
inputs          
p 
к " К          з
E__inference_dropout_63_layer_call_and_return_conditional_losses_42892^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ з
E__inference_dropout_63_layer_call_and_return_conditional_losses_42897^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ 
*__inference_dropout_63_layer_call_fn_42902Q4в1
*в'
!К
inputs         А
p
к "К         А
*__inference_dropout_63_layer_call_fn_42907Q4в1
*в'
!К
inputs         А
p 
к "К         А╡
E__inference_dropout_64_layer_call_and_return_conditional_losses_41885l;в8
1в.
(К%
inputs         BB
p
к "-в*
#К 
0         BB
Ъ ╡
E__inference_dropout_64_layer_call_and_return_conditional_losses_41890l;в8
1в.
(К%
inputs         BB
p 
к "-в*
#К 
0         BB
Ъ Н
*__inference_dropout_64_layer_call_fn_41895_;в8
1в.
(К%
inputs         BB
p
к " К         BBН
*__inference_dropout_64_layer_call_fn_41900_;в8
1в.
(К%
inputs         BB
p 
к " К         BB╡
E__inference_dropout_65_layer_call_and_return_conditional_losses_42253l;в8
1в.
(К%
inputs         !! 
p
к "-в*
#К 
0         !! 
Ъ ╡
E__inference_dropout_65_layer_call_and_return_conditional_losses_42258l;в8
1в.
(К%
inputs         !! 
p 
к "-в*
#К 
0         !! 
Ъ Н
*__inference_dropout_65_layer_call_fn_42263_;в8
1в.
(К%
inputs         !! 
p
к " К         !! Н
*__inference_dropout_65_layer_call_fn_42268_;в8
1в.
(К%
inputs         !! 
p 
к " К         !! ╡
E__inference_dropout_66_layer_call_and_return_conditional_losses_42621l;в8
1в.
(К%
inputs          
p
к "-в*
#К 
0          
Ъ ╡
E__inference_dropout_66_layer_call_and_return_conditional_losses_42626l;в8
1в.
(К%
inputs          
p 
к "-в*
#К 
0          
Ъ Н
*__inference_dropout_66_layer_call_fn_42631_;в8
1в.
(К%
inputs          
p
к " К          Н
*__inference_dropout_66_layer_call_fn_42636_;в8
1в.
(К%
inputs          
p 
к " К          з
E__inference_dropout_67_layer_call_and_return_conditional_losses_42919^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ з
E__inference_dropout_67_layer_call_and_return_conditional_losses_42924^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ 
*__inference_dropout_67_layer_call_fn_42929Q4в1
*в'
!К
inputs         А
p
к "К         А
*__inference_dropout_67_layer_call_fn_42934Q4в1
*в'
!К
inputs         А
p 
к "К         А│
C__inference_face_net_layer_call_and_return_conditional_losses_39991ыX;<56STUVJKLMqrklЙКЛМАБВГзибв┐└┴┬╢╖╕╣хц▀р А¤■ЎўЇїУФНОBв?
8в5
+К(
input_6         ╞╞
p

 
к "KвH
AЪ>
К
0/0         
К
0/1         
Ъ │
C__inference_face_net_layer_call_and_return_conditional_losses_40144ыX;<56STUVJKLMqrklЙКЛМАБВГзибв┐└┴┬╢╖╕╣хц▀рА¤ ■ўЇЎїУФНОBв?
8в5
+К(
input_6         ╞╞
p 

 
к "KвH
AЪ>
К
0/0         
К
0/1         
Ъ ▓
C__inference_face_net_layer_call_and_return_conditional_losses_41102ъX;<56STUVJKLMqrklЙКЛМАБВГзибв┐└┴┬╢╖╕╣хц▀р А¤■ЎўЇїУФНОAв>
7в4
*К'
inputs         ╞╞
p

 
к "KвH
AЪ>
К
0/0         
К
0/1         
Ъ ▓
C__inference_face_net_layer_call_and_return_conditional_losses_41310ъX;<56STUVJKLMqrklЙКЛМАБВГзибв┐└┴┬╢╖╕╣хц▀рА¤ ■ўЇЎїУФНОAв>
7в4
*К'
inputs         ╞╞
p 

 
к "KвH
AЪ>
К
0/0         
К
0/1         
Ъ К
(__inference_face_net_layer_call_fn_40409▌X;<56STUVJKLMqrklЙКЛМАБВГзибв┐└┴┬╢╖╕╣хц▀р А¤■ЎўЇїУФНОBв?
8в5
+К(
input_6         ╞╞
p

 
к "=Ъ:
К
0         
К
1         К
(__inference_face_net_layer_call_fn_40673▌X;<56STUVJKLMqrklЙКЛМАБВГзибв┐└┴┬╢╖╕╣хц▀рА¤ ■ўЇЎїУФНОBв?
8в5
+К(
input_6         ╞╞
p 

 
к "=Ъ:
К
0         
К
1         Й
(__inference_face_net_layer_call_fn_41421▄X;<56STUVJKLMqrklЙКЛМАБВГзибв┐└┴┬╢╖╕╣хц▀р А¤■ЎўЇїУФНОAв>
7в4
*К'
inputs         ╞╞
p

 
к "=Ъ:
К
0         
К
1         Й
(__inference_face_net_layer_call_fn_41532▄X;<56STUVJKLMqrklЙКЛМАБВГзибв┐└┴┬╢╖╕╣хц▀рА¤ ■ўЇЎїУФНОAв>
7в4
*К'
inputs         ╞╞
p 

 
к "=Ъ:
К
0         
К
1         к
E__inference_flatten_15_layer_call_and_return_conditional_losses_42642a7в4
-в*
(К%
inputs          
к "&в#
К
0         А@
Ъ В
*__inference_flatten_15_layer_call_fn_42647T7в4
-в*
(К%
inputs          
к "К         А@к
E__inference_flatten_16_layer_call_and_return_conditional_losses_42653a7в4
-в*
(К%
inputs          
к "&в#
К
0         А@
Ъ В
*__inference_flatten_16_layer_call_fn_42658T7в4
-в*
(К%
inputs          
к "К         А@д
H__inference_gender_output_layer_call_and_return_conditional_losses_42986X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ |
-__inference_gender_output_layer_call_fn_42991K/в,
%в"
 К
inputs         
к "К         ю
K__inference_max_pooling2d_45_layer_call_and_return_conditional_losses_38052ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_45_layer_call_fn_38058СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_46_layer_call_and_return_conditional_losses_38284ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_46_layer_call_fn_38290СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_47_layer_call_and_return_conditional_losses_38516ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_47_layer_call_fn_38522СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_48_layer_call_and_return_conditional_losses_38064ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_48_layer_call_fn_38070СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_49_layer_call_and_return_conditional_losses_38296ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_49_layer_call_fn_38302СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ю
K__inference_max_pooling2d_50_layer_call_and_return_conditional_losses_38528ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ╞
0__inference_max_pooling2d_50_layer_call_fn_38534СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╝
#__inference_signature_wrapper_40794ФX;<56STUVJKLMqrklЙКЛМАБВГзибв┐└┴┬╢╖╕╣хц▀рА¤ ■ўЇЎїУФНОEвB
в 
;к8
6
input_6+К(
input_6         ╞╞"qкn
2

age_output$К!

age_output         
8
gender_output'К$
gender_output         