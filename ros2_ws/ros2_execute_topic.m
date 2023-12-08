
%receive command from python node and execute on real robot

%% ros1 
%% init ros
setenv('ROS_MASTER_URI', 'http://192.168.1.10:11311')
setenv('ROS_IP', '192.168.1.44')
rosshutdown()
rosinit()
robot = Sawyer();
joint_sub = rossubscriber('/robot/joint_states', 'DataFormat','struct');

pubJointCmd = rospublisher('/robot/limb/right/joint_command')
msgJointCmd = rosmessage(pubJointCmd);

pubGripper = rospublisher('/gripper_command')

msgGripper =  rosmessage(pubGripper)


%% ros2 subscribe
node = ros2node("/ros2_matlab");
pause(5)
% sub= ros2subscriber(node,"/hello_py");
sub= ros2subscriber(node,"/inference");
pause(5)


%% tmp: store data from the python bc_offline node
abc=[];
while 1  
    [msg,status,statustext] = receive(sub,10);
    pos = msg.data;
    if length(pos) < 6
        continue
    end
    pos=double(pos);
    abc=cat(2, abc, pos);
end
% abc=abc';


%% [ensure safety] receive ros2 data from python bc_offline/live node and execute on robot.
th = .2;
dt = 1/5;
robot.setJointsMsg(joint_sub.receive());
robot.plotObject()
msgJointCmd.Names = robot.jointNames; %([1 3:8]);
msgJointCmd.Mode = msgJointCmd.VELOCITYMODE;
figure(1)
[msg,status,statustext] = receive(sub,10);
while 1
    tic
    robot.setJointsMsg(joint_sub.LatestMessage);

    % [msg,status,statustext] = receive(sub,10);
    msg = sub.LatestMessage;
    pos = msg.data;
    if length(pos) < 6
        continue
    end
    curT = robot.getBodyTransform('right_electric_gripper_base');
    eul = rotm2eul(curT(1:3, 1:3))';
    curPos = cat(1, curT(1:3,end), eul);

    error = zeros(6, 1);
    error(1:3) = 1.5*(pos(1:3) - curPos(1:3)); 

    curR = curT(1:3, 1:3);
    refR = eul2rotm(pos(4:6)', 'ZYX');
    errorR = (refR*curR');
    % [V, D] = eig(errorR);
    % errorAxis = real(V(:, end));
    axisAngle = rotm2axang(errorR);
    errorAxis = axisAngle(1:3);
    angle = acos((trace(errorR) -1)/2);

    error(4:6) = .5*errorAxis*angle;

    % error

    % robot.setJointsMsg(joint_sub.LatestMessage);
    Jall = robot.getJacobian();
    J = Jall(73:73+5, :);

    jointVel = pinv(J)*error;

    %ensure vel isn't too high. parameter to play=th
    if max(abs(jointVel) > th)
        jointVel = th*jointVel./max(abs(jointVel));
    end

    msgJointCmd.Velocity = jointVel;
    send(pubJointCmd, msgJointCmd);
    
    % sim
    % q = robot.getJoints();
    % qNew = q + jointVel*dt;
    % robot.setJoints(qNew)
    % robot.plotObject();
    % pause(dt-toc);
    toc
end


%% [plot only] receive ros2 data from python bc_offline 
th = .1;
dt = 1/5;
robot.setJointsMsg(joint_sub.receive());
robot.plotObject()
msgJointCmd.Names = robot.jointNames; %([1 3:8]);
msgJointCmd.Mode = msgJointCmd.VELOCITYMODE;
figure(1)
while 1
    tic
    robot.setJointsMsg(joint_sub.receive());

    [msg,status,statustext] = receive(sub,10);
    pos = msg.data;
    if length(pos) < 6
        continue
    end
    curT = robot.getBodyTransform('right_electric_gripper_base');
    eul = rotm2eul(curT(1:3, 1:3))';
    curPos = cat(1, curT(1:3,end), eul);

    error = zeros(6, 1);
    error(1:3) = 0.5*(pos(1:3) - curPos(1:3)); 

    curR = curT(1:3, 1:3);
    refR = eul2rotm(pos(4:6)', 'ZYX');
    errorR = (refR*curR');
    % [V, D] = eig(errorR);
    % errorAxis = real(V(:, end));
    axisAngle = rotm2axang(errorR);
    errorAxis = axisAngle(1:3);
    angle = acos((trace(errorR) -1)/2);

    error(4:6) = .2*errorAxis*angle;

    % error

    % robot.setJointsMsg(joint_sub.LatestMessage);
    Jall = robot.getJacobian();
    J = Jall(73:73+5, :);

    jointVel = pinv(J)*error;

    %ensure vel isn't too high. parameter to play=th
    if max(abs(jointVel) > th)
        jointVel = th*jointVel./max(abs(jointVel));
    end
 
    % sim
    q = robot.getJoints();
    qNew = q + jointVel*dt;
    robot.setJoints(qNew)
    % robot.plotObject();
    % pause(dt-toc);
    toc
end


